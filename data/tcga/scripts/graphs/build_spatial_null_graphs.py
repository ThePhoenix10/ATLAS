#!/usr/bin/env python3

import argparse
import json
import math
import zlib
from pathlib import Path
import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

DEFAULT_INPUT_DIR = '/workspace/data/tcga/spatial-aware'
DEFAULT_OUTPUT_DIR = '/workspace/data/tcga/spatial-null'
DEFAULT_SEED = 42
DEFAULT_RADIUS_MULTIPLIER = 1.5

def patient_seed(patient_id, base_seed):
    crc = zlib.crc32(str(patient_id).encode('utf-8'))

    return (int(base_seed) + int(crc)) % 2 ** 32

def to_numpy_1d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)

def to_numpy_2d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)

    if arr.ndim != 2:
        raise RuntimeError(f'Expected 2D array, got shape {arr.shape}')
    return arr

def get_scalar(graph, key, default=None):
    if key not in graph:
        return default
    value = graph[key]

    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value
    return value

def build_radius_graph_from_positions(pos, tile_stride, radius_multiplier):
    n = int(pos.shape[0])

    if n == 0:
        raise RuntimeError('Graph has zero nodes.')
    if n == 1:
        edge_index = np.zeros((2, 0), dtype=np.int32)
        edge_attr = np.zeros((0, 3), dtype=np.float32)

        return (edge_index, edge_attr)
    radius = float(tile_stride) * float(radius_multiplier)
    tree = cKDTree(pos.astype(np.float64))
    undirected_pairs = sorted(tree.query_pairs(r=radius, output_type='set'))
    src_list = []
    dst_list = []
    attr_list = []

    for i, j in undirected_pairs:
        dx_ij = float(pos[j, 0] - pos[i, 0])
        dy_ij = float(pos[j, 1] - pos[i, 1])
        dist_ij = math.sqrt(dx_ij * dx_ij + dy_ij * dy_ij)
        norm_dist = dist_ij / float(tile_stride)
        norm_dx = dx_ij / float(tile_stride)
        norm_dy = dy_ij / float(tile_stride)
        src_list.append(i)
        dst_list.append(j)
        attr_list.append([norm_dist, norm_dx, norm_dy])
        src_list.append(j)
        dst_list.append(i)
        attr_list.append([norm_dist, -norm_dx, -norm_dy])
    edge_index = np.asarray([src_list, dst_list], dtype=np.int32)
    edge_attr = np.asarray(attr_list, dtype=np.float32)

    if edge_index.size == 0:
        edge_index = np.zeros((2, 0), dtype=np.int32)
    if edge_attr.size == 0:
        edge_attr = np.zeros((0, 3), dtype=np.float32)
    return (edge_index, edge_attr)

def shuffle_tile_placements(input_path, output_path, base_seed):
    graph = torch.load(input_path, map_location='cpu', weights_only=False)
    required = ['node_indices']
    missing = [k for k in required if k not in graph]

    if missing:
        raise RuntimeError(f'{input_path.name} missing required keys: {missing}')
    patient_id = str(graph.get('patient_id', input_path.stem))
    node_indices = to_numpy_1d(graph['node_indices']).astype(np.int64)
    n = int(len(node_indices))

    if 'pos' in graph:
        pos = to_numpy_2d(graph['pos']).astype(np.float32)
    else:
        source_h5 = None

        for key in ['source_h5', 'h5_path', 'source_h5_path', 'embedding_path', 'feature_path']:
            if key in graph:
                source_h5 = graph[key]
                break
        if source_h5 is None:
            raise RuntimeError(f'{input_path.name}: graph has no pos and no source H5 path.')
        source_h5 = str(source_h5)

        with h5py.File(source_h5, 'r') as f:
            if 'coords_patching' in f:
                coord_key = 'coords_patching'
            elif 'coords' in f:
                coord_key = 'coords'
            else:
                raise RuntimeError(f'{input_path.name}: neither coords_patching nor coords exists in {source_h5}')
            coord_ds = f[coord_key]

            if coord_ds.ndim == 3:
                if coord_ds.shape[0] != 1:
                    raise RuntimeError(f'{input_path.name}: unexpected coordinate shape {coord_ds.shape} in {source_h5}')
                order = np.argsort(node_indices)
                sorted_idx = node_indices[order]
                sorted_pos = np.asarray(coord_ds[0, sorted_idx, :], dtype=np.float32)
            elif coord_ds.ndim == 2:
                order = np.argsort(node_indices)
                sorted_idx = node_indices[order]
                sorted_pos = np.asarray(coord_ds[sorted_idx, :], dtype=np.float32)
            else:
                raise RuntimeError(f'{input_path.name}: unexpected coordinate shape {coord_ds.shape} in {source_h5}')
            reverse = np.empty_like(order)
            reverse[order] = np.arange(len(order))
            pos = sorted_pos[reverse]
    if int(pos.shape[0]) != n:
        raise RuntimeError(f'{input_path.name}: node_indices has {n} rows but pos has {pos.shape[0]} rows.')
    if 'tile_stride' in graph:
        tile_stride = float(get_scalar(graph, 'tile_stride'))
    elif n <= 1:
        tile_stride = 1.0
    else:
        tree = cKDTree(pos)
        distances, _ = tree.query(pos, k=min(2, n))

        if distances.ndim == 2 and distances.shape[1] >= 2:
            nearest = distances[:, 1]
            nearest = nearest[np.isfinite(nearest) & (nearest > 0)]

            if len(nearest) == 0:
                tile_stride = 1.0
            else:
                tile_stride = float(np.median(nearest))
        else:
            tile_stride = 1.0
    radius_multiplier = float(get_scalar(graph, 'radius_multiplier', DEFAULT_RADIUS_MULTIPLIER))

    if n <= 1:
        shuffled_pos = pos.copy()
        permutation = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(patient_seed(patient_id, base_seed))
        permutation = rng.permutation(n)

        if np.array_equal(permutation, np.arange(n)):
            permutation = np.roll(permutation, 1)
        shuffled_pos = pos[permutation].copy()
    edge_index, edge_attr = build_radius_graph_from_positions(shuffled_pos, tile_stride=tile_stride, radius_multiplier=radius_multiplier)
    out = dict(graph)

    if 'pos' in graph and torch.is_tensor(graph['pos']):
        out['pos'] = torch.as_tensor(shuffled_pos, dtype=graph['pos'].dtype)
    elif 'pos' in graph:
        out['pos'] = shuffled_pos.astype(np.asarray(graph['pos']).dtype, copy=False)
    else:
        out['pos'] = torch.as_tensor(shuffled_pos, dtype=torch.float32)
    out['tile_stride'] = float(tile_stride)

    if torch.is_tensor(graph['edge_index']):
        out['edge_index'] = torch.as_tensor(edge_index, dtype=graph['edge_index'].dtype)
    else:
        out['edge_index'] = edge_index
    if torch.is_tensor(graph['edge_attr']):
        out['edge_attr'] = torch.as_tensor(edge_attr, dtype=graph['edge_attr'].dtype)
    else:
        out['edge_attr'] = edge_attr
    out['spatial_control'] = 'within_slide_tile_placement_permutation'
    out['spatial_control_seed'] = int(base_seed)
    out['spatial_control_patient_seed'] = int(patient_seed(patient_id, base_seed))
    out['spatial_control_method'] = 'Each tile kept its original UNI2-h embedding identity (node_indices unchanged), but tile coordinates were randomly permuted across nodes within the same patient. A new radius-neighborhood graph was then rebuilt from the permuted coordinates, and edge_attr [distance, dx, dy] was recomputed. This destroys true spatial tile placement while preserving morphology content, node count, patient identity, and the same radius-graph construction rule.'
    out['spatial_control_changed_coordinates'] = True
    out['spatial_control_changed_edges'] = True
    out['spatial_control_changed_edge_attr'] = True
    out['spatial_control_changed_node_indices'] = False
    out['spatial_control_permutation'] = torch.as_tensor(permutation, dtype=torch.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    changed_fraction = float(np.mean(np.any(shuffled_pos != pos, axis=1))) if n > 0 else 0.0

    return {'patient_id': patient_id, 'num_nodes': n, 'num_edges': int(edge_index.shape[1]), 'changed_fraction_coordinates': changed_fraction, 'tile_stride': tile_stride, 'radius_multiplier': radius_multiplier}

def main():
    parser = argparse.ArgumentParser(description='Create spatial-destruction control graphs by keeping each tile embedding with its tile while randomly permuting tile placements (coordinates) within each patient and rebuilding the radius graph.')
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    graph_paths = sorted(input_dir.glob('*.pt'))

    if not graph_paths:
        raise RuntimeError(f'No .pt graphs found in {input_dir}')
    if args.limit is not None:
        graph_paths = graph_paths[:args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 100)
    print('SPATIAL DESTRUCTION CONTROL')
    print('WITHIN-SLIDE TILE-PLACEMENT PERMUTATION')
    print('=' * 100)
    print(f'Input graph directory:  {input_dir}')
    print(f'Output graph directory: {output_dir}')
    print(f'Graphs selected:         {len(graph_paths):,}')
    print(f'Base seed:               {args.seed}')
    print()
    print('KEPT:')
    print('  patient identity')
    print('  tile embedding identity (node_indices)')
    print('  source H5')
    print('  number of nodes')
    print('  same radius-graph rule')
    print()
    print('PERTURBED:')
    print('  tile coordinates / placement')
    print('  edge_index (rebuilt from new coordinates)')
    print('  edge_attr (recomputed distance / dx / dy)')
    print('=' * 100)
    successful = 0
    skipped = 0
    failed = []
    total_nodes = 0
    total_edges = 0
    changed_fractions = []

    for input_path in tqdm(graph_paths, desc='Creating shuffled-placement graphs', unit='graph'):
        output_path = output_dir / input_path.name

        if output_path.exists() and (not args.overwrite):
            skipped += 1
            continue
        try:
            info = shuffle_tile_placements(input_path=input_path, output_path=output_path, base_seed=args.seed)
            successful += 1
            total_nodes += info['num_nodes']
            total_edges += info['num_edges']
            changed_fractions.append(info['changed_fraction_coordinates'])
        except Exception as exc:
            failed.append({'file': input_path.name, 'error': str(exc)})
    metadata = {'control': 'within_slide_tile_placement_permutation', 'base_seed': int(args.seed), 'input_dir': str(input_dir), 'output_dir': str(output_dir), 'selected': int(len(graph_paths)), 'successful': int(successful), 'skipped': int(skipped), 'failed': int(len(failed)), 'total_nodes_processed': int(total_nodes), 'total_edges_processed': int(total_edges), 'mean_fraction_coordinates_reassigned': float(np.mean(changed_fractions)) if changed_fractions else None, 'preserved': ['patient identity', 'source H5', 'tile embedding identity', 'node_indices', 'node count', 'radius-graph construction rule'], 'perturbed': ['tile coordinates / placement', 'edge_index', 'edge_attr', 'spatial neighborhood relationships'], 'method': 'Within each patient, tile coordinates were randomly permuted across nodes while node_indices were kept unchanged. A new radius graph was rebuilt from the permuted coordinates, and edge_attr [distance, dx, dy] was recomputed. This keeps each UNI2-h embedding attached to its original tile while destroying the true placement of tiles in tissue space.', 'failures': failed}
    metadata_path = output_dir / 'shuffle_metadata.json'

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print()
    print('=' * 100)
    print('SPATIAL CONTROL CREATION COMPLETE')
    print('=' * 100)
    print(f'Successful: {successful:,}')
    print(f'Skipped:    {skipped:,}')
    print(f'Failed:     {len(failed):,}')

    if changed_fractions:
        print(f'Mean fraction of tiles moved to a new coordinate: {np.mean(changed_fractions):.6f}')
    print(f'Metadata:   {metadata_path}')
    print('=' * 100)

    if failed:
        raise RuntimeError(f'{len(failed)} graphs failed. See {metadata_path}')

if __name__ == '__main__':
    main()
