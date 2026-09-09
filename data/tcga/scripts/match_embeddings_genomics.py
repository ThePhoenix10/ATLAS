#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
import pandas as pd
import requests

DEFAULT_GENOMIC_PREFIX = '/workspace/data/tcga/genomics/fused/patient_fused_intogen_filtered'
DEFAULT_PATHOLOGY_SPLITS_DIR = '/workspace/splits'
DEFAULT_PATHOLOGY_LABELS = '/workspace/metadata/tcga_labels.csv'
DEFAULT_GRAPH_DIR = '/workspace/data/tcga_spatial_graphs'
DEFAULT_OUTPUT_ROOT = '/workspace/data/tcga/multimodal'
GDC_CASES_ENDPOINT = 'https://api.gdc.cancer.gov/cases'

def parse_args():
    parser = argparse.ArgumentParser(description="Build the matched TCGA pathology+genomics cohort. The genomic row identifiers are '<tumor label> TAB <GDC case UUID>'. This script maps GDC case UUIDs to TCGA submitter IDs, links each patient to the existing pathology graph, and preserves the EXACT original pathology-only site-stratified 5-fold split assignments.")
    parser.add_argument('--genomic-prefix', default=DEFAULT_GENOMIC_PREFIX)
    parser.add_argument('--pathology-splits-dir', default=DEFAULT_PATHOLOGY_SPLITS_DIR)
    parser.add_argument('--pathology-labels', default=DEFAULT_PATHOLOGY_LABELS)
    parser.add_argument('--graph-dir', default=DEFAULT_GRAPH_DIR)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--gdc-batch-size', type=int, default=400)
    parser.add_argument('--gdc-timeout', type=int, default=120)
    parser.add_argument('--gdc-retries', type=int, default=5)
    parser.add_argument('--refresh-gdc-map', action='store_true', help='Ignore cached UUID->TCGA mapping and query GDC again.')

    return parser.parse_args()

def normalize_tcga_patient_id(value):
    value = str(value).strip().upper()

    if not value:
        return value
    parts = value.split('-')

    if len(parts) >= 3 and parts[0] == 'TCGA':
        return '-'.join(parts[:3])
    return value

def parse_genomic_row_line(line, row_index):
    line = line.rstrip('\n')

    if not line.strip():
        raise RuntimeError(f'Blank genomic row identifier at row {row_index}')
    if '\t' not in line:
        raise RuntimeError(f"Expected '<tumor label> TAB <GDC UUID>' in genomic row {row_index}, but found: {line!r}")
    tumor_label, case_uuid = line.rsplit('\t', 1)
    tumor_label = tumor_label.strip()
    case_uuid = case_uuid.strip().lower()

    if not tumor_label:
        raise RuntimeError(f'Missing tumor label at genomic row {row_index}')
    if not case_uuid:
        raise RuntimeError(f'Missing GDC case UUID at genomic row {row_index}')
    return (tumor_label, case_uuid)

def load_genomic_rows(genomic_prefix):
    genomic_prefix = str(genomic_prefix)
    rows_path = Path(genomic_prefix + '_rows.txt')
    matrix_path = Path(genomic_prefix + '.npz')
    columns_path = Path(genomic_prefix + '_columns.txt')

    for path in (rows_path, matrix_path, columns_path):
        if not path.exists():
            raise FileNotFoundError(path)
    parsed_rows = []

    with open(rows_path, 'r') as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            tumor_label, case_uuid = parse_genomic_row_line(line, row_index)
            parsed_rows.append({'genomic_row_index': int(row_index), 'genomic_tumor_label': tumor_label, 'gdc_case_uuid': case_uuid})
    df = pd.DataFrame(parsed_rows)

    if df.empty:
        raise RuntimeError(f'No genomic rows loaded from {rows_path}')
    duplicate_uuid = df[df['gdc_case_uuid'].duplicated(keep=False)]

    if not duplicate_uuid.empty:
        raise RuntimeError('Duplicate GDC case UUIDs in genomic row file:\n' + duplicate_uuid.head(30).to_string(index=False))
    return (df, rows_path, matrix_path, columns_path)

def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]

def query_gdc_batch(case_uuids, timeout, retries):
    filters = {'op': 'in', 'content': {'field': 'case_id', 'value': list(case_uuids)}}
    payload = {'filters': json.dumps(filters), 'fields': 'case_id,submitter_id,project.project_id', 'format': 'JSON', 'size': str(len(case_uuids) + 10)}
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(GDC_CASES_ENDPOINT, data=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            hits = data.get('data', {}).get('hits', [])
            records = []

            for hit in hits:
                case_id = str(hit.get('case_id', '')).strip().lower()
                submitter_id = normalize_tcga_patient_id(hit.get('submitter_id', ''))
                project = hit.get('project') or {}
                project_id = str(project.get('project_id', '')).strip()

                if case_id and submitter_id:
                    records.append({'gdc_case_uuid': case_id, 'patient_norm': submitter_id, 'gdc_project_id': project_id})
            return records
        except Exception as exc:
            last_error = exc

            if attempt < retries:
                wait_seconds = min(2 ** (attempt - 1), 15)
                print(f'  GDC request failed (attempt {attempt}/{retries}): {exc}')
                print(f'  Retrying in {wait_seconds}s...')
                time.sleep(wait_seconds)
    raise RuntimeError(f'GDC request failed after {retries} attempts: {last_error}')

def build_or_load_gdc_mapping(genomic_rows_df, cache_path, batch_size, timeout, retries, refresh):
    requested = set(genomic_rows_df['gdc_case_uuid'])
    cached_df = None

    if cache_path.exists() and (not refresh):
        cached_df = pd.read_csv(cache_path)
        required = {'gdc_case_uuid', 'patient_norm'}

        if required.issubset(cached_df.columns):
            cached_df = cached_df.copy()
            cached_df['gdc_case_uuid'] = cached_df['gdc_case_uuid'].astype(str).str.strip().str.lower()
            cached_df['patient_norm'] = cached_df['patient_norm'].map(normalize_tcga_patient_id)
            cached_set = set(cached_df['gdc_case_uuid'])
            missing = sorted(requested - cached_set)

            if not missing:
                print(f'Using cached GDC UUID mapping: {cache_path}')

                return cached_df[cached_df['gdc_case_uuid'].isin(requested)].copy()
            print(f'Cached GDC map is missing {len(missing)} UUIDs; querying only missing cases.')
        else:
            print(f'Cached mapping exists but lacks required columns; rebuilding: {cache_path}')
            cached_df = None
    if cached_df is None or refresh:
        cached_df = pd.DataFrame(columns=['gdc_case_uuid', 'patient_norm', 'gdc_project_id'])
        to_query = sorted(requested)
    else:
        cached_set = set(cached_df['gdc_case_uuid'])
        to_query = sorted(requested - cached_set)
    new_records = []
    total_batches = (len(to_query) + batch_size - 1) // batch_size

    for batch_number, batch in enumerate(chunks(to_query, batch_size), start=1):
        print(f'Querying GDC UUID mapping batch {batch_number}/{total_batches} ({len(batch)} cases)...')
        batch_records = query_gdc_batch(batch, timeout=timeout, retries=retries)
        new_records.extend(batch_records)
    if new_records:
        new_df = pd.DataFrame(new_records)
        combined = pd.concat([cached_df, new_df], ignore_index=True)
    else:
        combined = cached_df.copy()
    combined = combined.drop_duplicates(subset=['gdc_case_uuid'], keep='last').sort_values('gdc_case_uuid').reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path, index=False)
    found = set(combined['gdc_case_uuid'])
    missing = sorted(requested - found)

    if missing:
        missing_path = cache_path.parent / 'gdc_case_uuids_not_resolved.txt'

        with open(missing_path, 'w') as f:
            for value in missing:
                f.write(value + '\n')
        raise RuntimeError(f'GDC did not resolve {len(missing)} of {len(requested)} case UUIDs. Missing UUIDs written to {missing_path}')
    return combined[combined['gdc_case_uuid'].isin(requested)].copy()

def load_labels(labels_csv):
    df = pd.read_csv(labels_csv)
    required = {'patient_norm', 'label_text', 'site'}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f'Pathology labels file missing columns: {sorted(missing)}')
    df = df.copy()
    df['patient_norm'] = df['patient_norm'].map(normalize_tcga_patient_id)
    duplicate = df[df['patient_norm'].duplicated(keep=False)]

    if not duplicate.empty:
        raise RuntimeError('Duplicate patients in pathology labels:\n' + duplicate.head(30).to_string(index=False))
    return df

def discover_graphs(graph_dir):
    graph_dir = Path(graph_dir)
    graph_map = {}

    for path in sorted(graph_dir.glob('*.pt')):
        patient = normalize_tcga_patient_id(path.stem)

        if patient in graph_map:
            raise RuntimeError(f'Duplicate graph patient detected: {patient}\nExisting: {graph_map[patient]}\nNew:      {path}')
        graph_map[patient] = str(path.resolve())
    if not graph_map:
        raise RuntimeError(f'No .pt graphs found in {graph_dir}')
    return graph_map

def build_master_manifest(genomic_rows_df, gdc_mapping_df, labels_df, graph_map, genomic_matrix_path):
    merged = genomic_rows_df.merge(gdc_mapping_df, on='gdc_case_uuid', how='left', validate='one_to_one')

    if merged['patient_norm'].isna().any():
        bad = merged[merged['patient_norm'].isna()]

        raise RuntimeError(f'{len(bad)} genomic cases remain without a TCGA submitter ID.')
    duplicate_patient = merged[merged['patient_norm'].duplicated(keep=False)]

    if not duplicate_patient.empty:
        raise RuntimeError('Multiple genomic rows map to the same TCGA patient:\n' + duplicate_patient.head(30).to_string(index=False))
    label_columns = ['patient_norm', 'label_text', 'site']

    if 'project_id' in labels_df.columns:
        label_columns.append('project_id')
    merged = merged.merge(labels_df[label_columns], on='patient_norm', how='left', validate='one_to_one')
    missing_label = merged[merged['label_text'].isna()]

    if not missing_label.empty:
        raise RuntimeError(f'{len(missing_label)} genomic patients are missing from pathology labels. Examples:\n' + missing_label[['gdc_case_uuid', 'patient_norm', 'genomic_tumor_label']].head(30).to_string(index=False))
    merged['graph_path'] = merged['patient_norm'].map(graph_map)
    missing_graph = merged[merged['graph_path'].isna()]

    if not missing_graph.empty:
        raise RuntimeError(f'{len(missing_graph)} matched patients are missing pathology graphs. Examples:\n' + missing_graph[['patient_norm', 'gdc_case_uuid']].head(30).to_string(index=False))
    merged['genomic_matrix_path'] = str(genomic_matrix_path)
    ordered_columns = ['patient_norm', 'gdc_case_uuid', 'genomic_tumor_label', 'label_text']

    if 'project_id' in merged.columns:
        ordered_columns.append('project_id')
    ordered_columns.extend(['site', 'graph_path', 'genomic_row_index', 'genomic_matrix_path', 'gdc_project_id'])
    merged = merged[ordered_columns].sort_values('genomic_row_index').reset_index(drop=True)

    return merged

def load_original_split(path):
    df = pd.read_csv(path)
    required = {'patient_norm', 'split'}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f'{path} missing columns: {sorted(missing)}')
    df = df.copy()
    df['patient_norm'] = df['patient_norm'].map(normalize_tcga_patient_id)
    assignments = df[['patient_norm', 'split']].drop_duplicates(subset=['patient_norm'])

    return assignments

def create_fold(fold, original_split_path, manifest, output_splits_dir):
    original = load_original_split(original_split_path)
    assignment_map = dict(zip(original['patient_norm'], original['split']))
    fold_df = manifest.copy()
    fold_df['split'] = fold_df['patient_norm'].map(assignment_map)
    missing = fold_df[fold_df['split'].isna()]

    if not missing.empty:
        raise RuntimeError(f'Fold {fold}: {len(missing)} matched patients are absent from the original pathology-only split.')
    valid = {'train', 'val', 'test'}
    observed = set(fold_df['split'].unique())

    if not observed.issubset(valid):
        raise RuntimeError(f'Fold {fold}: unexpected split values {sorted(observed - valid)}')
    verify = fold_df[['patient_norm', 'split']].merge(original, on='patient_norm', how='left', suffixes=('_matched', '_original'), validate='one_to_one')
    mismatches = verify[verify['split_matched'] != verify['split_original']]

    if not mismatches.empty:
        raise RuntimeError(f'Fold {fold}: {len(mismatches)} assignments differ from the original pathology-only split.')
    fold_dir = output_splits_dir / f'fold_{fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    split_path = fold_dir / 'split.csv'
    fold_df.to_csv(split_path, index=False)
    class_counts = fold_df.groupby(['label_text', 'split']).size().unstack(fill_value=0).reset_index()
    class_counts.to_csv(fold_dir / 'class_counts.csv', index=False)
    site_counts = fold_df.groupby(['site', 'split']).size().unstack(fill_value=0).reset_index()
    site_counts.to_csv(fold_dir / 'site_counts.csv', index=False)
    counts = fold_df['split'].value_counts()

    return {'fold': int(fold), 'train_n': int(counts.get('train', 0)), 'val_n': int(counts.get('val', 0)), 'test_n': int(counts.get('test', 0)), 'verified_same_assignment_n': int(len(fold_df)), 'source_split': str(original_split_path), 'output_split': str(split_path)}

def main():
    args = parse_args()
    output_root = Path(args.output_root)
    metadata_dir = output_root / 'metadata'
    splits_dir = output_root / 'splits'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 110)
    print('BUILDING MATCHED PATHOLOGY + GENOMICS DATASET')
    print('=' * 110)
    genomic_rows_df, genomic_rows_path, genomic_matrix_path, genomic_columns_path = load_genomic_rows(args.genomic_prefix)
    print(f'Genomic rows:   {len(genomic_rows_df)}')
    print(f'Genomic matrix: {genomic_matrix_path}')
    print('Row format:     <tumor label> TAB <GDC case UUID>')
    print()
    print('=' * 110)
    print('MAPPING GDC CASE UUIDs -> TCGA PATIENT BARCODES')
    print('=' * 110)
    mapping_cache = metadata_dir / 'gdc_case_uuid_to_tcga_barcode.csv'
    gdc_mapping_df = build_or_load_gdc_mapping(genomic_rows_df=genomic_rows_df, cache_path=mapping_cache, batch_size=args.gdc_batch_size, timeout=args.gdc_timeout, retries=args.gdc_retries, refresh=args.refresh_gdc_map)
    print(f'Resolved UUIDs: {len(gdc_mapping_df)}')
    print(f'Mapping cache:  {mapping_cache}')
    print()
    print('=' * 110)
    print('LOADING PATHOLOGY LABELS AND GRAPHS')
    print('=' * 110)
    labels_df = load_labels(args.pathology_labels)
    graph_map = discover_graphs(args.graph_dir)
    print(f'Pathology labels: {len(labels_df)}')
    print(f'Pathology graphs: {len(graph_map)}')
    manifest = build_master_manifest(genomic_rows_df=genomic_rows_df, gdc_mapping_df=gdc_mapping_df, labels_df=labels_df, graph_map=graph_map, genomic_matrix_path=genomic_matrix_path)
    manifest_path = metadata_dir / 'matched_patients.csv'
    manifest.to_csv(manifest_path, index=False)
    print()
    print('=' * 110)
    print('MATCHED COHORT')
    print('=' * 110)
    print(f'Matched patients: {len(manifest)}')
    print(f"Tumor origins:    {manifest['label_text'].nunique()}")
    print(f"Sites:            {manifest['site'].nunique()}")
    print(f'Manifest:         {manifest_path}')

    if len(manifest) != 8402:
        raise RuntimeError(f'Expected exactly 8402 matched genomic patients, but obtained {len(manifest)}.')
    print()
    print('=' * 110)
    print('REUSING EXACT ORIGINAL SITE-STRATIFIED 5-FOLD ASSIGNMENTS')
    print('=' * 110)
    original_splits_dir = Path(args.pathology_splits_dir)
    fold_summaries = []

    for fold in range(1, 6):
        original_split_path = original_splits_dir / f'fold_{fold}' / 'split.csv'

        if not original_split_path.exists():
            raise FileNotFoundError(original_split_path)
        summary = create_fold(fold=fold, original_split_path=original_split_path, manifest=manifest, output_splits_dir=splits_dir)
        fold_summaries.append(summary)
        print(f"Fold {fold}: train={summary['train_n']} val={summary['val_n']} test={summary['test_n']} | assignment match={summary['verified_same_assignment_n']}/{len(manifest)}")
    pd.DataFrame(fold_summaries).to_csv(splits_dir / 'fold_summary.csv', index=False)
    overall_class_counts = manifest['label_text'].value_counts().rename_axis('label_text').reset_index(name='n')
    overall_class_counts.to_csv(metadata_dir / 'overall_class_counts.csv', index=False)
    overall_site_counts = manifest['site'].value_counts().rename_axis('site').reset_index(name='n')
    overall_site_counts.to_csv(metadata_dir / 'overall_site_counts.csv', index=False)
    metadata = {'matched_patients': int(len(manifest)), 'tumor_origins': int(manifest['label_text'].nunique()), 'sites': int(manifest['site'].nunique()), 'genomic_matrix': str(genomic_matrix_path), 'genomic_rows': str(genomic_rows_path), 'genomic_columns': str(genomic_columns_path), 'gdc_uuid_mapping': str(mapping_cache), 'pathology_labels': str(args.pathology_labels), 'pathology_graph_dir': str(args.graph_dir), 'original_pathology_splits': str(original_splits_dir), 'matched_manifest': str(manifest_path), 'split_strategy': 'Exact reuse of the original pathology-only site-stratified five-fold train/validation/test assignments, restricted to the matched 8402-patient pathology+genomics cohort.', 'split_assignments_recomputed': False, 'folds': fold_summaries}

    with open(metadata_dir / 'matched_dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print()
    print('=' * 110)
    print('DONE')
    print('=' * 110)
    print(f'Matched manifest: {manifest_path}')
    print(f'Matched splits:   {splits_dir}')
    print(f'GDC UUID map:     {mapping_cache}')
    print()
    print('IMPORTANT: split assignments were NOT regenerated.')
    print('Every one of the 8402 matched patients retains its original pathology-only fold/train/val/test assignment.')
    print('=' * 110)

if __name__ == '__main__':
    main()
