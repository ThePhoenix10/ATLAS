#!/usr/bin/env python3

import argparse
import ast
import json
import re
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

DEFAULT_EMBED_ROOT = '/workspace/data/cptac/embeddings'
DEFAULT_EMBED_MAP = '/workspace/data/cptac/metadata/case_id_embedding_map.csv'
DEFAULT_CPTAC_MATRIX = '/workspace/data/cptac/genomics/fused/cptac_matrix.npz'
DEFAULT_CPTAC_ROWS = '/workspace/data/cptac/genomics/fused/cptac_matrix_rows.txt'
DEFAULT_CPTAC_MUT_MATRIX = '/workspace/data/cptac/genomics/mutations/patient_mutation_matrix.npz'
DEFAULT_CPTAC_MUT_ROWS = '/workspace/data/cptac/genomics/mutations/patient_mutation_matrix_rows.txt'
DEFAULT_CPTAC_MUT_COLUMNS = '/workspace/data/cptac/genomics/mutations/patient_mutation_matrix_columns.txt'
DEFAULT_CPTAC_CNA_MATRIX = '/workspace/data/cptac/genomics/cna/patient_cna_matrix.npz'
DEFAULT_CPTAC_CNA_ROWS = '/workspace/data/cptac/genomics/cna/patient_cna_matrix_rows.txt'
DEFAULT_CPTAC_CNA_COLUMNS = '/workspace/data/cptac/genomics/cna/patient_cna_matrix_columns.txt'
DEFAULT_CPTAC_SBS_MATRIX = '/workspace/data/cptac/genomics/sbs96/patient_signature_matrix.npz'
DEFAULT_CPTAC_SBS_ROWS = '/workspace/data/cptac/genomics/sbs96/patient_signature_matrix_rows.txt'
DEFAULT_CPTAC_SBS_COLUMNS = '/workspace/data/cptac/genomics/sbs96/patient_signature_matrix_columns.txt'
DEFAULT_ALIGNED_CPTAC_MATRIX = '/workspace/data/cptac/genomics/aligned/cptac_matrix.npz'
DEFAULT_TCGA_FILTERED_MATRIX = '/workspace/data/tcga/genomics/fused/patient_fused_intogen_filtered.npz'
DEFAULT_TCGA_FILTERED_COLUMNS = '/workspace/data/tcga/genomics/fused/patient_fused_intogen_filtered_columns.txt'
DEFAULT_OUTPUT_GRAPH_DIR = '/workspace/data/cptac/spatial-aware'
DEFAULT_OUTPUT_H5_DIR = '/workspace/data/cptac/patient_h5'
DEFAULT_OUTPUT_MANIFEST = '/workspace/data/cptac/metadata/external_validation_manifest.csv'
RADIUS_MULTIPLIER = 1.5
EXPECTED_EMBED_DIM = 1536

def normalize_id(x):
    return str(x).strip().upper()

def load_txt_lines(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]

def load_npz_matrix(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return sp.load_npz(path)
    except Exception:
        pass
    obj = np.load(path, allow_pickle=False)

    for key in ('matrix', 'X', 'arr_0', 'data'):
        if key in obj.files and obj[key].ndim == 2:
            return obj[key]
    mats = [obj[k] for k in obj.files if obj[k].ndim == 2]

    if len(mats) == 1:
        return mats[0]
    raise RuntimeError(f'Could not identify 2-D matrix inside {path}; keys={obj.files}')

def dense_row(matrix, idx):
    if sp.issparse(matrix):
        x = matrix.getrow(idx).toarray().reshape(-1)
    else:
        x = np.asarray(matrix[idx]).reshape(-1)
    return np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

def read_cptac_rows(path):
    lines = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    rows = []
    uuid_re = re.compile('^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')

    for line_number, line in enumerate(lines, start=1):
        if '\t' in line:
            parts = line.split('\t')
        elif ',' in line:
            parts = line.split(',')
        else:
            parts = line.split()
        parts = [x.strip() for x in parts if x.strip()]

        if not parts:
            continue
        lower = [x.lower() for x in parts]

        if 'case_id' in lower or 'patient_id' in lower or 'patient_norm' in lower:
            continue
        if len(parts) < 2:
            raise RuntimeError(f'Could not parse line {line_number} of {path}: {line!r}')
        tumor_origin = parts[0]
        case_id = parts[1]

        if uuid_re.match(parts[0]) and (not uuid_re.match(parts[1])):
            case_id = parts[0]
            tumor_origin = parts[1]
        if not uuid_re.match(case_id):
            raise RuntimeError(f'Expected CPTAC case UUID on line {line_number}, but got {case_id!r}. Full line: {line!r}')
        rows.append({'case_id': normalize_id(case_id), 'tumor_origin': tumor_origin})
    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f'No rows parsed from {path}')
    if df['case_id'].duplicated().any():
        dup = df.loc[df['case_id'].duplicated(keep=False), 'case_id'].tolist()

        raise RuntimeError(f'True duplicate CPTAC case UUIDs found after parsing. Examples: {dup[:10]}')
    print(f"Parsed CPTAC rows: {len(df):,} | unique case UUIDs: {df['case_id'].nunique():,} | tumor origins: {df['tumor_origin'].nunique():,}")

    return df

def detect_case_column(df):
    lower = {c.lower(): c for c in df.columns}

    for name in ('case_id', 'case', 'patient_id', 'patient', 'patient_norm', 'submitter_id'):
        if name in lower:
            return lower[name]
    for c in df.columns:
        lc = c.lower()

        if 'case' in lc or 'patient' in lc:
            return c
    raise RuntimeError(f'Could not identify case/patient column in embedding map. Columns={df.columns.tolist()}')

def split_path_field(value):
    if pd.isna(value):
        return []
    s = str(value).strip()

    if not s:
        return []
    if s.startswith('[') and s.endswith(']'):
        try:
            value = ast.literal_eval(s)

            if isinstance(value, (list, tuple)):
                return [str(x).strip() for x in value if str(x).strip()]
        except Exception:
            pass
    return [x.strip().strip('"\'') for x in re.split('[;|]', s) if x.strip()]

def resolve_h5(raw, embed_root):
    p = Path(raw)

    if p.exists():
        return p.resolve()
    p2 = Path(embed_root) / raw

    if p2.exists():
        return p2.resolve()
    matches = list(Path(embed_root).rglob(Path(raw).name))

    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(f'Ambiguous H5 basename: {Path(raw).name}')
    return None

def load_embedding_map(csv_path, embed_root):
    df = pd.read_csv(csv_path)
    case_col = detect_case_column(df)
    path_cols = [c for c in df.columns if c != case_col and any((token in c.lower() for token in ('h5', 'embedding', 'feature', 'path')))]

    if not path_cols:
        for c in df.columns:
            if c == case_col:
                continue
            vals = df[c].dropna().astype(str).head(50)

            if any(('.h5' in x.lower() or '.hdf5' in x.lower() for x in vals)):
                path_cols.append(c)
    if not path_cols:
        raise RuntimeError(f'No embedding path column found. Columns={df.columns.tolist()}')
    mapping = {}

    for _, row in df.iterrows():
        case_id = normalize_id(row[case_col])
        paths = []

        for col in path_cols:
            for raw in split_path_field(row[col]):
                if '.h5' not in raw.lower() and '.hdf5' not in raw.lower():
                    continue
                p = resolve_h5(raw, embed_root)

                if p is not None:
                    paths.append(p)
        if paths:
            mapping.setdefault(case_id, []).extend(paths)
    for case_id in mapping:
        mapping[case_id] = sorted(set(mapping[case_id]))
    return mapping

def load_features_coords(h5_path):
    with h5py.File(h5_path, 'r') as f:
        if 'features' not in f:
            raise KeyError(f"'features' missing from {h5_path}")
        x = np.asarray(f['features'])

        if x.ndim == 3:
            if x.shape[0] != 1:
                raise RuntimeError(f'Unexpected features shape {x.shape}')
            x = x[0]
        if x.ndim != 2 or x.shape[1] != EXPECTED_EMBED_DIM:
            raise RuntimeError(f'Expected [N,{EXPECTED_EMBED_DIM}] features in {h5_path}; got {x.shape}')
        coord_key = None

        for key in ('coords', 'coords_patching'):
            if key in f:
                coord_key = key
                break
        if coord_key is None:
            raise KeyError(f'Neither coords nor coords_patching exists in {h5_path}')
        coords = np.asarray(f[coord_key])

        if coords.ndim == 3:
            if coords.shape[0] != 1:
                raise RuntimeError(f'Unexpected coords shape {coords.shape}')
            coords = coords[0]
        coords = coords.reshape(-1, coords.shape[-1])[:, :2]

        if coords.shape[0] != x.shape[0]:
            raise RuntimeError(f'Feature/coordinate count mismatch in {h5_path}: {x.shape[0]} vs {coords.shape[0]}')
        return (x.astype(np.float32, copy=False), coords.astype(np.float64, copy=False))

def infer_stride(coords):
    if len(coords) < 2:
        return 1.0
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]

    if nearest.size:
        return float(np.median(nearest))
    candidates = []

    for axis in (0, 1):
        vals = np.sort(np.unique(coords[:, axis]))
        diff = np.diff(vals)
        diff = diff[diff > 0]

        if diff.size:
            candidates.append(np.min(diff))
    return float(np.median(candidates)) if candidates else 1.0

def build_edges(coords):
    if len(coords) <= 1:
        return (np.empty((2, 0), dtype=np.int64), np.empty((0, 3), dtype=np.float32), 1.0)
    stride = infer_stride(coords)

    if not np.isfinite(stride) or stride <= 0:
        raise RuntimeError(f'Invalid inferred stride {stride}')
    radius = RADIUS_MULTIPLIER * stride
    pairs = cKDTree(coords).query_pairs(r=radius, output_type='ndarray')

    if pairs.size == 0:
        return (np.empty((2, 0), dtype=np.int64), np.empty((0, 3), dtype=np.float32), stride)
    src = pairs[:, 0].astype(np.int64)
    dst = pairs[:, 1].astype(np.int64)
    delta = coords[dst] - coords[src]
    dx = delta[:, 0]
    dy = delta[:, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    fwd_attr = np.stack([dist / stride, dx / stride, dy / stride], axis=1).astype(np.float32)
    rev_attr = np.stack([dist / stride, -dx / stride, -dy / stride], axis=1).astype(np.float32)
    edge_index = np.concatenate([np.stack([src, dst], axis=0), np.stack([dst, src], axis=0)], axis=1)
    edge_attr = np.concatenate([fwd_attr, rev_attr], axis=0)

    return (edge_index, edge_attr, stride)

def build_patient_h5(out_path, slide_paths):
    feature_blocks = []
    coord_blocks = []
    edge_blocks = []
    edge_attr_blocks = []
    slide_offsets = []
    slide_strides = []
    slide_names = []
    node_offset = 0
    x_cursor = 0.0

    for slide_path in slide_paths:
        features, coords = load_features_coords(slide_path)
        edge_index, edge_attr, stride = build_edges(coords)

        if edge_index.shape[1]:
            edge_index = edge_index + node_offset
        shifted = coords.copy()

        if shifted.size:
            shifted[:, 0] = shifted[:, 0] - shifted[:, 0].min() + x_cursor
            x_cursor = shifted[:, 0].max() + 10.0 * stride
        feature_blocks.append(features)
        coord_blocks.append(shifted)
        edge_blocks.append(edge_index)
        edge_attr_blocks.append(edge_attr)
        slide_offsets.append((node_offset, node_offset + len(features)))
        slide_strides.append(float(stride))
        slide_names.append(str(slide_path))
        node_offset += len(features)
    features = np.concatenate(feature_blocks, axis=0)
    coords = np.concatenate(coord_blocks, axis=0)
    nonempty_ei = [x for x in edge_blocks if x.shape[1] > 0]
    nonempty_ea = [x for x in edge_attr_blocks if x.shape[0] > 0]

    if nonempty_ei:
        edge_index = np.concatenate(nonempty_ei, axis=1)
        edge_attr = np.concatenate(nonempty_ea, axis=0)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 3), dtype=np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('features', data=features, compression='gzip', compression_opts=1)
        f.create_dataset('coords', data=coords)
        f.create_dataset('coords_patching', data=coords)
        str_dtype = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('source_slides', data=np.asarray(slide_names, dtype=object), dtype=str_dtype)
        f.create_dataset('slide_offsets', data=np.asarray(slide_offsets, dtype=np.int64))
        f.create_dataset('slide_strides', data=np.asarray(slide_strides, dtype=np.float32))
    return (len(features), edge_index, edge_attr, slide_strides)
UUID_RE = re.compile('[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')

def read_uuid_row_ids(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)
    row_ids = []

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()

        if not line:
            continue
        match = UUID_RE.search(line)

        if match is None:
            lower = line.lower()

            if 'case_id' in lower or 'patient_id' in lower or 'patient_norm' in lower:
                continue
            raise RuntimeError(f'No CPTAC UUID found on line {line_number} of {path}: {line!r}')
        row_ids.append(normalize_id(match.group(0)))
    if not row_ids:
        raise RuntimeError(f'No UUID rows parsed from {path}')
    if len(set(row_ids)) != len(row_ids):
        raise RuntimeError(f'Duplicate UUIDs in modality row file: {path}')
    return row_ids

def build_aligned_cptac_matrix(target_case_ids, target_columns, mut_matrix_path, mut_rows_path, mut_columns_path, cna_matrix_path, cna_rows_path, cna_columns_path, sbs_matrix_path, sbs_rows_path, sbs_columns_path):
    print()
    print('=' * 90)
    print('BUILDING CPTAC MATRIX IN EXACT TCGA INTOGEN-FILTERED FEATURE SPACE')
    print('=' * 90)
    source_specs = {'mutation': (load_npz_matrix(mut_matrix_path), read_uuid_row_ids(mut_rows_path), load_txt_lines(mut_columns_path)), 'cna': (load_npz_matrix(cna_matrix_path), read_uuid_row_ids(cna_rows_path), load_txt_lines(cna_columns_path)), 'sbs': (load_npz_matrix(sbs_matrix_path), read_uuid_row_ids(sbs_rows_path), load_txt_lines(sbs_columns_path))}

    for name, (matrix, rows, columns) in source_specs.items():
        if matrix.shape[0] != len(rows):
            raise RuntimeError(f'{name} matrix/row mismatch: matrix={matrix.shape[0]}, rows={len(rows)}')
        if matrix.shape[1] != len(columns):
            raise RuntimeError(f'{name} matrix/column mismatch: matrix={matrix.shape[1]}, columns={len(columns)}')
        print(f'{name:8s} | matrix={matrix.shape} | rows={len(rows):,} | columns={len(columns):,}')
    target_case_ids = [normalize_id(x) for x in target_case_ids]
    target_row_index = {case_id: i for i, case_id in enumerate(target_case_ids)}
    aligned = np.zeros((len(target_case_ids), len(target_columns)), dtype=np.float32)
    target_col_index = {name: i for i, name in enumerate(target_columns)}
    mut_matrix, mut_rows, mut_columns = source_specs['mutation']
    mut_row_index = {case_id: i for i, case_id in enumerate(mut_rows)}
    mut_col_index = {name: i for i, name in enumerate(mut_columns)}
    target_mut = [c for c in target_columns if c.startswith('som_mut_')]
    matched_mut = [c for c in target_mut if c in mut_col_index]
    print(f'Mutation target columns matched: {len(matched_mut):,}/{len(target_mut):,}')
    cna_matrix, cna_rows, cna_columns = source_specs['cna']
    cna_row_index = {case_id: i for i, case_id in enumerate(cna_rows)}
    cna_col_index = {name: i for i, name in enumerate(cna_columns)}
    target_cna = [c for c in target_columns if c.startswith('cna_')]
    matched_cna = [c for c in target_cna if c in cna_col_index]
    print(f'CNA target columns matched: {len(matched_cna):,}/{len(target_cna):,}')
    sbs_matrix, sbs_rows, sbs_columns = source_specs['sbs']
    sbs_row_index = {case_id: i for i, case_id in enumerate(sbs_rows)}
    sbs_col_index = {name: i for i, name in enumerate(sbs_columns)}
    sbs_col_index_lower = {name.lower(): i for i, name in enumerate(sbs_columns)}
    target_sbs = [c for c in target_columns if not c.startswith('som_mut_') and (not c.startswith('cna_'))]
    sbs_mapping = {}

    for target_name in target_sbs:
        candidates = [target_name]

        if target_name.startswith('sig__'):
            candidates.append(target_name[len('sig__'):])
        else:
            candidates.append('sig__' + target_name)
        source_idx = None

        for candidate in candidates:
            if candidate in sbs_col_index:
                source_idx = sbs_col_index[candidate]
                break
            if candidate.lower() in sbs_col_index_lower:
                source_idx = sbs_col_index_lower[candidate.lower()]
                break
        if source_idx is not None:
            sbs_mapping[target_name] = source_idx
    print(f'SBS target columns matched: {len(sbs_mapping):,}/{len(target_sbs):,}')

    if len(matched_mut) != len(target_mut):
        missing = [c for c in target_mut if c not in mut_col_index]
        print(f'WARNING: mutation training columns absent from CPTAC raw matrix. They will remain zero. First examples: {missing[:10]}')
    if len(matched_cna) != len(target_cna):
        missing = [c for c in target_cna if c not in cna_col_index]
        print(f'WARNING: CNA training columns absent from CPTAC raw matrix. They will remain zero. First examples: {missing[:10]}')
    if len(sbs_mapping) != len(target_sbs):
        missing = [c for c in target_sbs if c not in sbs_mapping]

        raise RuntimeError(f'SBS training schema does not align to CPTAC SBS96 matrix. Missing {len(missing)} columns. Examples: {missing[:10]}')
    mut_target_idx = np.asarray([target_col_index[c] for c in matched_mut], dtype=np.int64)
    mut_source_idx = np.asarray([mut_col_index[c] for c in matched_mut], dtype=np.int64)
    cna_target_idx = np.asarray([target_col_index[c] for c in matched_cna], dtype=np.int64)
    cna_source_idx = np.asarray([cna_col_index[c] for c in matched_cna], dtype=np.int64)
    sbs_target_idx = np.asarray([target_col_index[c] for c in target_sbs], dtype=np.int64)
    sbs_source_idx = np.asarray([sbs_mapping[c] for c in target_sbs], dtype=np.int64)
    mut_present = 0
    cna_present = 0
    sbs_present = 0

    for out_i, case_id in enumerate(target_case_ids):
        if case_id in mut_row_index:
            source_row = dense_row(mut_matrix, mut_row_index[case_id])
            aligned[out_i, mut_target_idx] = source_row[mut_source_idx]
            mut_present += 1
        if case_id in cna_row_index:
            source_row = dense_row(cna_matrix, cna_row_index[case_id])
            aligned[out_i, cna_target_idx] = source_row[cna_source_idx]
            cna_present += 1
        if case_id in sbs_row_index:
            source_row = dense_row(sbs_matrix, sbs_row_index[case_id])
            aligned[out_i, sbs_target_idx] = source_row[sbs_source_idx]
            sbs_present += 1
    print()
    print(f'Patient modality coverage | mutation={mut_present:,}/{len(target_case_ids):,} | CNA={cna_present:,}/{len(target_case_ids):,} | SBS={sbs_present:,}/{len(target_case_ids):,}')

    if aligned.shape[1] != len(target_columns):
        raise RuntimeError('Internal aligned matrix dimension error.')
    return aligned

def verify_training_schema(cptac_matrix, tcga_matrix_path, tcga_columns_path):
    tcga_columns = load_txt_lines(tcga_columns_path)
    tcga_matrix = load_npz_matrix(tcga_matrix_path)
    tcga_dim = tcga_matrix.shape[1]
    cptac_dim = cptac_matrix.shape[1]

    if tcga_dim != len(tcga_columns):
        raise RuntimeError(f'TCGA filtered matrix has {tcga_dim} features but column file has {len(tcga_columns)}')
    if cptac_dim != len(tcga_columns):
        raise RuntimeError(f'CPTAC matrix does not match the exact TCGA IntOGen-filtered training schema: CPTAC={cptac_dim}, TCGA={len(tcga_columns)}')
    mutation = [c for c in tcga_columns if c.startswith('som_mut_')]
    cna = [c for c in tcga_columns if c.startswith('cna_')]
    sbs = [c for c in tcga_columns if not c.startswith('som_mut_') and (not c.startswith('cna_'))]
    print('=' * 90)
    print('TCGA INTOGEN-FILTERED TRAINING SCHEMA')
    print('=' * 90)
    print(f'TCGA filtered features: {len(tcga_columns)}')
    print(f'  mutation: {len(mutation)}')
    print(f'  CNA:      {len(cna)}')
    print(f'  SBS:      {len(sbs)}')
    print(f'CPTAC matrix shape: {cptac_matrix.shape}')
    print('PASS: CPTAC dimensionality matches exact TCGA filtered feature schema.')

    return tcga_columns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_root', default=DEFAULT_EMBED_ROOT)
    parser.add_argument('--embedding_map', default=DEFAULT_EMBED_MAP)
    parser.add_argument('--cptac_matrix', default=DEFAULT_CPTAC_MATRIX)
    parser.add_argument('--cptac_rows', default=DEFAULT_CPTAC_ROWS)
    parser.add_argument('--tcga_filtered_matrix', default=DEFAULT_TCGA_FILTERED_MATRIX)
    parser.add_argument('--tcga_filtered_columns', default=DEFAULT_TCGA_FILTERED_COLUMNS)
    parser.add_argument('--output_graph_dir', default=DEFAULT_OUTPUT_GRAPH_DIR)
    parser.add_argument('--output_h5_dir', default=DEFAULT_OUTPUT_H5_DIR)
    parser.add_argument('--output_manifest', default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    cptac_rows = read_cptac_rows(args.cptac_rows)
    tcga_columns = load_txt_lines(args.tcga_filtered_columns)
    tcga_matrix = load_npz_matrix(args.tcga_filtered_matrix)

    if tcga_matrix.shape[1] != len(tcga_columns):
        raise RuntimeError(f'TCGA filtered matrix/column mismatch: matrix={tcga_matrix.shape}, columns={len(tcga_columns)}')
    cptac_matrix = load_npz_matrix(args.cptac_matrix)

    if cptac_matrix.shape[0] != len(cptac_rows):
        raise RuntimeError(
            f'CPTAC row mismatch: matrix={cptac_matrix.shape[0]}, '
            f'rows={len(cptac_rows)}'
        )

    if cptac_matrix.shape[1] != len(tcga_columns):
        raise RuntimeError(
            f'CPTAC fused genomic width={cptac_matrix.shape[1]:,}, but '
            f'TCGA training width={len(tcga_columns):,}. Run '
            f'fuse_cptac_genomics.py with the TCGA training columns file '
            f'before building CPTAC graphs.'
        )

    print()
    print(f'Loaded fused CPTAC genomic matrix: {cptac_matrix.shape}')
    verify_training_schema(cptac_matrix, args.tcga_filtered_matrix, args.tcga_filtered_columns)
    embedding_map = load_embedding_map(args.embedding_map, args.embed_root)
    graph_dir = Path(args.output_graph_dir)
    h5_dir = Path(args.output_h5_dir)
    manifest_path = Path(args.output_manifest)
    graph_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    print()
    print('=' * 90)
    print('BUILDING CPTAC ATLAS PATIENT GRAPHS')
    print('=' * 90)
    print(f'CPTAC rows: {len(cptac_rows):,}')
    print(f'Embedding-mapped cases: {len(embedding_map):,}')
    print(f'Radius rule: {RADIUS_MULTIPLIER} x slide-specific stride')
    print('Edge attrs: [distance/stride, dx/stride, dy/stride]')
    print('Multiple slides: merged per patient, but NO edges cross slide boundaries.')
    records = []

    for row_idx, row in cptac_rows.iterrows():
        case_id = row['case_id']
        tumor_origin = row['tumor_origin']
        slides = embedding_map.get(case_id, [])
        graph_path = graph_dir / f'{case_id}.pt'
        patient_h5_path = h5_dir / f'{case_id}.h5'

        if not slides:
            print(f'[MISS] {case_id}: no embedding')
            records.append({'case_id': case_id, 'tumor_origin': tumor_origin, 'status': 'missing_embedding', 'graph_path': '', 'patient_h5_path': '', 'n_slides': 0, 'n_nodes': 0, 'n_edges': 0, 'genomic_dim': cptac_matrix.shape[1]})
            continue
        if graph_path.exists() and patient_h5_path.exists() and (not args.overwrite):
            graph = torch.load(graph_path, map_location='cpu', weights_only=False)
            print(f'[KEEP] {case_id}: existing')
            records.append({'case_id': case_id, 'tumor_origin': tumor_origin, 'status': 'existing', 'graph_path': str(graph_path), 'patient_h5_path': str(patient_h5_path), 'n_slides': len(slides), 'n_nodes': len(graph['node_indices']), 'n_edges': graph['edge_index'].shape[1], 'genomic_dim': graph['genomic_x'].numel()})
            continue
        try:
            n_nodes, edge_index, edge_attr, slide_strides = build_patient_h5(patient_h5_path, slides)
            genomic_x = dense_row(cptac_matrix, row_idx)

            if len(genomic_x) != len(tcga_columns):
                raise RuntimeError(f'{case_id}: genomic dim {len(genomic_x)} != training dim {len(tcga_columns)}')
            graph = {'patient_id': case_id, 'case_id': case_id, 'label_text': tumor_origin, 'source_h5': str(patient_h5_path.resolve()), 'source_slides': [str(x) for x in slides], 'node_indices': torch.arange(n_nodes, dtype=torch.long), 'edge_index': torch.from_numpy(edge_index).long(), 'edge_attr': torch.from_numpy(edge_attr).float(), 'genomic_x': torch.from_numpy(genomic_x).float(), 'radius_multiplier': RADIUS_MULTIPLIER, 'slide_strides': slide_strides, 'genomic_schema': 'exact_TCGA_IntOGen_filtered_training_columns', 'genomic_columns_path': str(Path(args.tcga_filtered_columns).resolve())}
            torch.save(graph, graph_path)
            print(f'[OK] {case_id} | slides={len(slides)} | nodes={n_nodes:,} | edges={edge_index.shape[1]:,}')
            records.append({'case_id': case_id, 'tumor_origin': tumor_origin, 'status': 'built', 'graph_path': str(graph_path), 'patient_h5_path': str(patient_h5_path), 'n_slides': len(slides), 'n_nodes': n_nodes, 'n_edges': edge_index.shape[1], 'genomic_dim': len(genomic_x)})
        except Exception as exc:
            print(f'[FAIL] {case_id}: {exc}')
            records.append({'case_id': case_id, 'tumor_origin': tumor_origin, 'status': 'failed', 'error': repr(exc), 'graph_path': '', 'patient_h5_path': '', 'n_slides': len(slides), 'n_nodes': 0, 'n_edges': 0, 'genomic_dim': cptac_matrix.shape[1]})
    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    summary = {'total_cptac_rows': len(cptac_rows), 'built': int((manifest['status'] == 'built').sum()), 'existing': int((manifest['status'] == 'existing').sum()), 'missing_embedding': int((manifest['status'] == 'missing_embedding').sum()), 'failed': int((manifest['status'] == 'failed').sum()), 'training_feature_count': len(tcga_columns), 'radius_multiplier': RADIUS_MULTIPLIER, 'graph_dir': str(graph_dir), 'patient_h5_dir': str(h5_dir), 'manifest': str(manifest_path)}
    summary_path = manifest_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2))
    print()
    print('=' * 90)
    print('DONE')
    print('=' * 90)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
