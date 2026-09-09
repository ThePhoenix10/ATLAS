#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.spatial import cKDTree

DEFAULT_MANIFEST = '/workspace/data/tcga/multimodal/metadata/matched_patients.csv'
DEFAULT_H5_DIR = '/workspace/data/tcga/embeddings'
DEFAULT_GENOMIC_MATRIX = '/workspace/data/tcga/genomics/fused/patient_fused_intogen_filtered.npz'
DEFAULT_GENOMIC_COLUMNS = '/workspace/data/tcga/genomics/fused/patient_fused_intogen_filtered_columns.txt'
DEFAULT_OUTPUT_DIR = '/workspace/data/tcga/spatial-aware'

def parse_args():
    parser = argparse.ArgumentParser(description='Rebuild the matched pathology+genomics TCGA graph dataset directly from UNI2-h H5 embeddings and the IntOGen-filtered genomic matrix. This does NOT depend on previously created pathology graph .pt files.')
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--h5-dir', default=DEFAULT_H5_DIR)
    parser.add_argument('--genomic-matrix', default=DEFAULT_GENOMIC_MATRIX)
    parser.add_argument('--genomic-columns', default=DEFAULT_GENOMIC_COLUMNS)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--radius-multiplier', type=float, default=1.5)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--store-node-features', action='store_true', help='Store UNI2-h node features inside each .pt file. By default, graphs remain lightweight and store source_h5 + node_indices so features can be loaded from H5 during training.')

    return parser.parse_args()

def read_lines(path):
    with open(path, 'r') as f:
        return [line.rstrip('\n') for line in f if line.strip()]

def normalize_patient_id(value):
    value = str(value).strip().upper()
    parts = value.split('-')

    if len(parts) >= 3 and parts[0] == 'TCGA':
        return '-'.join(parts[:3])
    return value

def build_h5_index(h5_dir):
    h5_dir = Path(h5_dir)
    files = sorted(list(h5_dir.glob('*.h5')) + list(h5_dir.glob('*.hdf5')))

    if not files:
        raise RuntimeError(f'No H5 files found in {h5_dir}')
    mapping = {}

    for path in files:
        name = path.stem.upper()
        patient = None

        if 'TCGA-' in name:
            start = name.index('TCGA-')
            candidate = name[start:]
            parts = candidate.split('-')

            if len(parts) >= 3:
                patient = '-'.join(parts[:3])
        if patient is None:
            continue
        if patient in mapping:
            raise RuntimeError(f'More than one H5 found for {patient}:\n{mapping[patient]}\n{path}')
        mapping[patient] = str(path.resolve())
    return mapping

def squeeze_first_axis(array):
    arr = np.asarray(array)

    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr

def read_h5_arrays(path):
    with h5py.File(path, 'r') as f:
        if 'coords_patching' in f:
            coords = np.asarray(f['coords_patching'])
        elif 'coords' in f:
            coords = squeeze_first_axis(f['coords'])
        else:
            raise RuntimeError(f'{path}: no coords_patching or coords dataset')
        if 'features' not in f:
            raise RuntimeError(f'{path}: no features dataset')
        features = squeeze_first_axis(f['features'])
        attrs = {}

        for key in f.attrs.keys():
            value = f.attrs[key]

            try:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                elif hasattr(value, 'item'):
                    value = value.item()
            except Exception:
                pass
            attrs[str(key)] = value
    coords = np.asarray(coords, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise RuntimeError(f'{path}: coords shape must be [N,2], got {coords.shape}')
    if features.ndim != 2:
        raise RuntimeError(f'{path}: features shape must be [N,D], got {features.shape}')
    if coords.shape[0] != features.shape[0]:
        raise RuntimeError(f'{path}: coordinate/feature count mismatch {coords.shape[0]} vs {features.shape[0]}')
    return (coords, features, attrs)

def deduplicate_coordinates(coords, features):
    if len(coords) == 0:
        return (coords, features, np.empty(0, dtype=np.int64), 0)
    _, first_indices, inverse, counts = np.unique(coords, axis=0, return_index=True, return_inverse=True, return_counts=True)
    first_indices = np.sort(first_indices)
    duplicate_count = int(len(coords) - len(first_indices))

    if duplicate_count > 0:
        duplicate_groups = np.where(counts > 1)[0]

        for group_id in duplicate_groups:
            member_indices = np.where(inverse == group_id)[0]
            reference = features[member_indices[0]]

            if not np.allclose(features[member_indices], reference, rtol=1e-05, atol=1e-06):
                raise RuntimeError('Duplicate coordinates contain non-identical embeddings; refusing to drop duplicates.')
    return (coords[first_indices], features[first_indices], first_indices.astype(np.int64), duplicate_count)

def candidate_stride_from_attrs(attrs):
    preferred_keys = ['patch_size_level0', 'patch_size', 'stride', 'patch_stride']

    for key in preferred_keys:
        if key not in attrs:
            continue
        value = attrs[key]

        try:
            value = float(value)
        except Exception:
            continue
        if value > 0:
            return (value, f'h5_attr:{key}')
    return (None, None)

def infer_stride_from_coords(coords):
    if len(coords) < 2:
        raise RuntimeError('Cannot infer slide stride from fewer than 2 coordinates.')
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]

    if len(nearest) == 0:
        raise RuntimeError('Could not infer positive nearest-neighbor stride.')
    stride = float(np.median(nearest))

    return (stride, 'median_positive_nearest_neighbor')

def determine_stride(coords, attrs):
    stride, source = candidate_stride_from_attrs(attrs)

    if stride is None:
        stride, source = infer_stride_from_coords(coords)
    return (float(stride), source)

def build_fixed_radius_edges(coords, stride, radius_multiplier):
    radius = float(radius_multiplier * stride)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')

    if pairs.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 3), dtype=np.float32)

        return (edge_index, edge_attr, radius)
    src = pairs[:, 0]
    dst = pairs[:, 1]
    dx = coords[dst, 0] - coords[src, 0]
    dy = coords[dst, 1] - coords[src, 1]
    distance = np.sqrt(dx * dx + dy * dy)
    forward_attr = np.stack([distance / stride, dx / stride, dy / stride], axis=1).astype(np.float32)
    backward_attr = np.stack([distance / stride, -dx / stride, -dy / stride], axis=1).astype(np.float32)
    bidirectional_src = np.concatenate([src, dst])
    bidirectional_dst = np.concatenate([dst, src])
    edge_index = np.stack([bidirectional_src, bidirectional_dst], axis=0).astype(np.int64)
    edge_attr = np.concatenate([forward_attr, backward_attr], axis=0)

    return (edge_index, edge_attr, radius)

def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    genomic_matrix_path = Path(args.genomic_matrix)
    genomic_columns_path = Path(args.genomic_columns)
    output_dir = Path(args.output_dir)

    for path in (manifest_path, genomic_matrix_path, genomic_columns_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 110)
    print('REBUILDING MATCHED PATHOLOGY + GENOMICS SPATIAL GRAPHS FROM SOURCE DATA')
    print('=' * 110)
    manifest = pd.read_csv(manifest_path)
    required_columns = {'patient_norm', 'label_text', 'site', 'genomic_row_index'}
    missing = required_columns - set(manifest.columns)

    if missing:
        raise RuntimeError(f'Manifest missing columns: {sorted(missing)}')
    genomic_matrix = sparse.load_npz(genomic_matrix_path).tocsr()
    genomic_columns = read_lines(genomic_columns_path)

    if genomic_matrix.shape[0] != len(manifest):
        raise RuntimeError(f'Genomic matrix rows={genomic_matrix.shape[0]} but manifest rows={len(manifest)}')
    if genomic_matrix.shape[1] != len(genomic_columns):
        raise RuntimeError(f'Genomic matrix columns={genomic_matrix.shape[1]} but column-name count={len(genomic_columns)}')
    print(f'Matched patients:      {len(manifest)}')
    print(f'Genomic matrix:        {genomic_matrix.shape}')
    print(f'UNI2-h source folder:  {args.h5_dir}')
    print(f'Output directory:      {output_dir}')
    print(f'Radius multiplier:     {args.radius_multiplier}')
    print(f'Store node features:   {args.store_node_features}')
    print()
    print('Indexing UNI2-h H5 files...')
    h5_map = build_h5_index(args.h5_dir)
    print(f'Indexed H5 patients:   {len(h5_map)}')
    missing_h5 = []
    created = 0
    skipped = 0
    total_duplicates_removed = 0
    total_nodes = 0
    total_edges = 0
    stride_sources = {}
    stride_values = {}
    qc_rows = []

    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        patient = normalize_patient_id(row.patient_norm)
        source_h5 = h5_map.get(patient)

        if source_h5 is None:
            missing_h5.append(patient)
            continue
        output_path = output_dir / f'{patient}.pt'

        if output_path.exists() and (not args.overwrite):
            skipped += 1

            if position % 100 == 0 or position == len(manifest):
                print(f'{position}/{len(manifest)} | created={created} skipped={skipped}')
            continue
        coords, features, attrs = read_h5_arrays(source_h5)
        coords, features, node_indices, duplicate_count = deduplicate_coordinates(coords, features)
        stride, stride_source = determine_stride(coords, attrs)
        edge_index, edge_attr, radius = build_fixed_radius_edges(coords, stride, args.radius_multiplier)
        genomic_row_index = int(row.genomic_row_index)

        if genomic_row_index < 0 or genomic_row_index >= genomic_matrix.shape[0]:
            raise RuntimeError(f'{patient}: invalid genomic_row_index {genomic_row_index}')
        genomic_sparse = genomic_matrix.getrow(genomic_row_index)
        genomic_dense = genomic_sparse.toarray().reshape(-1).astype(np.float32)
        graph = {'patient_norm': patient, 'label_text': str(row.label_text), 'site': str(row.site), 'coords': torch.from_numpy(coords.astype(np.float32)), 'edge_index': torch.from_numpy(edge_index).long(), 'edge_attr': torch.from_numpy(edge_attr).float(), 'stride': float(stride), 'stride_source': stride_source, 'radius_multiplier': float(args.radius_multiplier), 'radius': float(radius), 'source_h5': str(Path(source_h5).resolve()), 'node_indices': torch.from_numpy(node_indices).long(), 'feature_dim': int(features.shape[1]), 'genomic_x': torch.from_numpy(genomic_dense).float(), 'genomic_feature_dim': int(genomic_dense.shape[0]), 'genomic_row_index': int(genomic_row_index), 'genomic_matrix_source': str(genomic_matrix_path.resolve()), 'genomic_columns_source': str(genomic_columns_path.resolve()), 'genomic_scope': 'patient-level global graph context'}

        if args.store_node_features:
            graph['x'] = torch.from_numpy(features).float()
        torch.save(graph, output_path)
        created += 1
        total_duplicates_removed += int(duplicate_count)
        total_nodes += int(coords.shape[0])
        total_edges += int(edge_index.shape[1])
        stride_sources[stride_source] = stride_sources.get(stride_source, 0) + 1
        stride_key = str(int(round(stride)))
        stride_values[stride_key] = stride_values.get(stride_key, 0) + 1
        qc_rows.append({'patient_norm': patient, 'source_h5': source_h5, 'graph_path': str(output_path), 'nodes': int(coords.shape[0]), 'directed_edges': int(edge_index.shape[1]), 'duplicates_removed': int(duplicate_count), 'stride': float(stride), 'stride_source': stride_source, 'radius': float(radius), 'feature_dim': int(features.shape[1]), 'genomic_feature_dim': int(genomic_dense.shape[0]), 'genomic_nonzero': int(genomic_sparse.nnz)})

        if position % 100 == 0 or position == len(manifest):
            print(f'{position}/{len(manifest)} | created={created} skipped={skipped} | nodes={coords.shape[0]} edges={edge_index.shape[1]} stride={stride:g}')
    if missing_h5:
        missing_path = output_dir / 'missing_h5_patients.txt'

        with open(missing_path, 'w') as f:
            for patient in missing_h5:
                f.write(patient + '\n')
        raise RuntimeError(f'{len(missing_h5)} patients have no source H5. See {missing_path}')
    graph_files = sorted(output_dir.glob('TCGA-*.pt'))

    if len(graph_files) != len(manifest):
        raise RuntimeError(f'Expected {len(manifest)} graph files but found {len(graph_files)}')
    qc_path = output_dir / 'graph_qc.csv'

    if qc_rows:
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    summary = {'matched_patients': int(len(manifest)), 'graph_files': int(len(graph_files)), 'created_this_run': int(created), 'skipped_existing': int(skipped), 'pathology_embedding_dim': 1536, 'genomic_feature_dim': int(genomic_matrix.shape[1]), 'radius_multiplier': float(args.radius_multiplier), 'total_duplicates_removed_this_run': int(total_duplicates_removed), 'total_nodes_this_run': int(total_nodes), 'total_directed_edges_this_run': int(total_edges), 'stride_sources_this_run': stride_sources, 'stride_values_this_run': stride_values, 'node_feature_storage': 'embedded in PT' if args.store_node_features else 'lightweight: source_h5 + node_indices; training loader reads UNI2-h rows from H5', 'genomics_storage': 'genomic_x stored directly in each graph as a patient-level global vector', 'source_h5_dir': str(Path(args.h5_dir).resolve()), 'genomic_matrix_source': str(genomic_matrix_path.resolve()), 'output_dir': str(output_dir.resolve())}
    summary_path = output_dir / 'graph_build_summary.json'

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print()
    print('=' * 110)
    print('DONE')
    print('=' * 110)
    print(f'Graphs:                    {len(graph_files)}')
    print(f'Expected matched patients: {len(manifest)}')
    print(f'Genomic feature dim:       {genomic_matrix.shape[1]}')
    print(f'Radius multiplier:         {args.radius_multiplier}')
    print(f'Output:                    {output_dir}')
    print(f'Summary:                   {summary_path}')
    print('=' * 110)

if __name__ == '__main__':
    main()
