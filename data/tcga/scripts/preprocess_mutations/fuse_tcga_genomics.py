#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
import numpy as np
from scipy import sparse

DEFAULT_MUTATION_PREFIX = '/workspace/data/tcga/genomics/mutations/patient_mutation_matrix'
DEFAULT_CNA_PREFIX = '/workspace/data/tcga/genomics/cna/patient_cna_matrix'
DEFAULT_SBS_PREFIX = '/workspace/data/tcga/genomics/sbs96/patient_signature_matrix'
DEFAULT_INTOGEN_DIR = '/workspace/data/reference/intogen'
DEFAULT_OUTPUT_DIR = '/workspace/data/tcga/genomics/fused'

def parse_args():
    parser = argparse.ArgumentParser(description='Assemble TCGA somatic mutation + CNA + SBS matrices by inner-joining patients, then filter mutation/CNA features to IntOGen cancer-driver genes while retaining all 96 SBS features.')
    parser.add_argument('--mutation-prefix', default=DEFAULT_MUTATION_PREFIX)
    parser.add_argument('--cna-prefix', default=DEFAULT_CNA_PREFIX)
    parser.add_argument('--sbs-prefix', default=DEFAULT_SBS_PREFIX)
    parser.add_argument('--intogen-dir', default=DEFAULT_INTOGEN_DIR)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--compendium-name', default='Compendium_Cancer_Genes.tsv', help='Filtered IntOGen driver-gene catalog to use.')

    return parser.parse_args()

def load_lines(path):
    with open(path, 'r') as f:
        return [line.rstrip('\n') for line in f]

def load_sparse_matrix(prefix):
    prefix = Path(prefix)
    npz_path = prefix.with_suffix('.npz')
    rows_path = Path(str(prefix) + '_rows.txt')
    cols_path = Path(str(prefix) + '_columns.txt')

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    if not cols_path.exists():
        raise FileNotFoundError(cols_path)
    matrix = sparse.load_npz(npz_path).tocsr()
    rows = load_lines(rows_path)
    cols = load_lines(cols_path)

    if matrix.shape[0] != len(rows):
        raise RuntimeError(f'Row count mismatch for {prefix}: matrix={matrix.shape[0]}, rows={len(rows)}')
    if matrix.shape[1] != len(cols):
        raise RuntimeError(f'Column count mismatch for {prefix}: matrix={matrix.shape[1]}, cols={len(cols)}')
    return (matrix, rows, cols)

def find_intogen_compendium(intogen_dir, preferred_name):
    intogen_dir = Path(intogen_dir)
    preferred = intogen_dir / preferred_name

    if preferred.exists():
        return preferred
    candidates = [intogen_dir / 'drivers.tsv', intogen_dir / 'Compendium_Cancer_Genes.tsv', intogen_dir / 'compendium_cancer_genes.tsv']

    for path in candidates:
        if path.exists():
            return path
    tsvs = sorted(intogen_dir.glob('*.tsv'))

    for path in tsvs:
        name = path.name.lower()

        if 'compendium' in name or ('driver' in name and 'unfiltered' not in name):
            return path
    raise FileNotFoundError(f'Could not find filtered IntOGen driver catalog in {intogen_dir}')

def load_intogen_driver_genes(compendium_path):
    compendium_path = Path(compendium_path)
    genes = set()

    with open(compendium_path, 'r') as f:
        header = f.readline().rstrip('\n').split('\t')
        normalized = [x.strip().upper() for x in header]

        if 'SYMBOL' not in normalized:
            raise RuntimeError(f'Could not find SYMBOL column in {compendium_path}. Header={header}')
        symbol_idx = normalized.index('SYMBOL')

        for line in f:
            line = line.rstrip('\n')

            if not line:
                continue
            parts = line.split('\t')

            if symbol_idx >= len(parts):
                continue
            gene = parts[symbol_idx].strip()

            if gene:
                genes.add(gene.upper())
    if not genes:
        raise RuntimeError(f'No driver genes loaded from {compendium_path}')
    return genes

def mutation_feature_gene(feature):
    prefix = 'som_mut_'

    if not feature.startswith(prefix):
        return None
    rest = feature[len(prefix):]

    if rest.endswith('_any'):
        return rest[:-4].upper()
    classifications = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Splice_Site', 'Nonstop_Mutation', 'Translation_Start_Site', 'Silent']

    for classification in classifications:
        suffix = '_' + classification

        if rest.endswith(suffix):
            return rest[:-len(suffix)].upper()
    parts = rest.split('_')

    if not parts:
        return None
    return parts[0].upper()

def cna_feature_gene(feature):
    prefix = 'cna_'

    if not feature.startswith(prefix):
        return None
    rest = feature[len(prefix):]
    states = ['any', 'del', 'loss', 'gain', 'amp']

    for state in states:
        suffix = '_' + state

        if rest.endswith(suffix):
            return rest[:-len(suffix)].upper()
    parts = rest.split('_')

    if not parts:
        return None
    return parts[0].upper()

def build_patient_index(rows):
    index = {}

    for i, patient in enumerate(rows):
        patient = str(patient)

        if patient in index:
            raise RuntimeError(f'Duplicate patient ID detected: {patient}')
        index[patient] = i
    return index

def inner_join_patients(mut_rows, cna_rows, sbs_rows):
    mut_set = set(mut_rows)
    cna_set = set(cna_rows)
    sbs_set = set(sbs_rows)
    common = sorted(mut_set & cna_set & sbs_set)

    if not common:
        raise RuntimeError('No patients are shared across mutation, CNA, and SBS matrices.')
    return common

def select_rows(matrix, rows, common_patients):
    index = build_patient_index(rows)
    selected = [index[p] for p in common_patients]

    return matrix[selected]

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 110)
    print('LOADING RAW TCGA GENOMIC MATRICES')
    print('=' * 110)
    mutation_matrix, mutation_rows, mutation_cols = load_sparse_matrix(args.mutation_prefix)
    cna_matrix, cna_rows, cna_cols = load_sparse_matrix(args.cna_prefix)
    sbs_matrix, sbs_rows, sbs_cols = load_sparse_matrix(args.sbs_prefix)
    print(f'Mutation: {mutation_matrix.shape[0]} patients x {mutation_matrix.shape[1]} features')
    print(f'CNA:      {cna_matrix.shape[0]} patients x {cna_matrix.shape[1]} features')
    print(f'SBS:      {sbs_matrix.shape[0]} patients x {sbs_matrix.shape[1]} features')
    print()
    print('=' * 110)
    print('INNER-JOINING PATIENTS')
    print('=' * 110)
    common_patients = inner_join_patients(mutation_rows, cna_rows, sbs_rows)
    mutation_matrix = select_rows(mutation_matrix, mutation_rows, common_patients)
    cna_matrix = select_rows(cna_matrix, cna_rows, common_patients)
    sbs_matrix = select_rows(sbs_matrix, sbs_rows, common_patients)
    print(f'Matched patients across all 3 genomic modalities: {len(common_patients)}')
    print()
    print('=' * 110)
    print('LOADING INTOGEN DRIVER GENES')
    print('=' * 110)
    compendium_path = find_intogen_compendium(args.intogen_dir, args.compendium_name)
    driver_genes = load_intogen_driver_genes(compendium_path)
    print(f'IntOGen catalog: {compendium_path}')
    print(f'Unique IntOGen driver genes: {len(driver_genes)}')
    print()
    print('=' * 110)
    print('FILTERING MUTATION FEATURES')
    print('=' * 110)
    kept_mutation_indices = []
    kept_mutation_cols = []
    matched_mutation_genes = set()

    for i, feature in enumerate(mutation_cols):
        gene = mutation_feature_gene(feature)

        if gene is not None and gene in driver_genes:
            kept_mutation_indices.append(i)
            kept_mutation_cols.append(feature)
            matched_mutation_genes.add(gene)
    filtered_mutation = mutation_matrix[:, kept_mutation_indices].tocsr()
    print(f'Mutation features before: {len(mutation_cols)}')
    print(f'Mutation features after:  {len(kept_mutation_cols)}')
    print(f'IntOGen genes represented in mutation matrix: {len(matched_mutation_genes)}')
    print()
    print('=' * 110)
    print('FILTERING CNA FEATURES')
    print('=' * 110)
    kept_cna_indices = []
    kept_cna_cols = []
    matched_cna_genes = set()

    for i, feature in enumerate(cna_cols):
        gene = cna_feature_gene(feature)

        if gene is not None and gene in driver_genes:
            kept_cna_indices.append(i)
            kept_cna_cols.append(feature)
            matched_cna_genes.add(gene)
    filtered_cna = cna_matrix[:, kept_cna_indices].tocsr()
    print(f'CNA features before: {len(cna_cols)}')
    print(f'CNA features after:  {len(kept_cna_cols)}')
    print(f'IntOGen genes represented in CNA matrix: {len(matched_cna_genes)}')
    print()
    print('=' * 110)
    print('KEEPING ALL SBS FEATURES')
    print('=' * 110)
    filtered_sbs = sbs_matrix.tocsr()
    kept_sbs_cols = list(sbs_cols)
    print(f'SBS features retained: {len(kept_sbs_cols)}')
    print()
    print('=' * 110)
    print('ASSEMBLING FILTERED FUSED GENOMIC MATRIX')
    print('=' * 110)
    fused = sparse.hstack([filtered_mutation, filtered_cna, filtered_sbs], format='csr', dtype=np.float32)
    fused_cols = kept_mutation_cols + kept_cna_cols + kept_sbs_cols
    fused_prefix = output_dir / 'patient_fused_intogen_filtered'
    sparse.save_npz(Path(str(fused_prefix) + '.npz'), fused, compressed=True)

    with open(str(fused_prefix) + '_rows.txt', 'w') as f:
        for patient in common_patients:
            f.write(patient + '\n')
    with open(str(fused_prefix) + '_columns.txt', 'w') as f:
        for column in fused_cols:
            f.write(column + '\n')
    with open(output_dir / 'intogen_driver_genes_used.txt', 'w') as f:
        for gene in sorted(driver_genes):
            f.write(gene + '\n')
    summary = {'mutation_input_prefix': str(args.mutation_prefix), 'cna_input_prefix': str(args.cna_prefix), 'sbs_input_prefix': str(args.sbs_prefix), 'intogen_catalog': str(compendium_path), 'intogen_unique_driver_genes': int(len(driver_genes)), 'matched_patients': int(len(common_patients)), 'mutation_features_before': int(len(mutation_cols)), 'mutation_features_after': int(len(kept_mutation_cols)), 'cna_features_before': int(len(cna_cols)), 'cna_features_after': int(len(kept_cna_cols)), 'sbs_features_retained': int(len(kept_sbs_cols)), 'final_features': int(fused.shape[1]), 'final_shape': [int(fused.shape[0]), int(fused.shape[1])], 'mutation_driver_genes_represented': int(len(matched_mutation_genes)), 'cna_driver_genes_represented': int(len(matched_cna_genes)), 'output_prefix': str(fused_prefix)}

    with open(output_dir / 'intogen_filtered_genomics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print()
    print('=' * 110)
    print('DONE')
    print('=' * 110)
    print(f'Final matrix shape: {fused.shape[0]} patients x {fused.shape[1]} features')
    print(f'Output prefix: {fused_prefix}')
    print(f"Summary: {output_dir / 'intogen_filtered_genomics_summary.json'}")
    print('=' * 110)

if __name__ == '__main__':
    main()
