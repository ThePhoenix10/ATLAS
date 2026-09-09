#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz
from tqdm import tqdm

TCGA_COHORT_TO_TUMOR_ORIGIN = {'TCGA-BRCA': 'Breast', 'TCGA-LUAD': 'Lung', 'TCGA-LUSC': 'Lung', 'TCGA-GBM': 'Brain', 'TCGA-LGG': 'Brain', 'TCGA-KICH': 'Kidney', 'TCGA-KIRC': 'Kidney', 'TCGA-KIRP': 'Kidney', 'TCGA-UCEC': 'Uterus', 'TCGA-UCS': 'Uterus', 'TCGA-HNSC': 'Head and Neck', 'TCGA-THCA': 'Thyroid', 'TCGA-COAD': 'Colon', 'TCGA-PRAD': 'Prostate', 'TCGA-BLCA': 'Bladder', 'TCGA-STAD': 'Stomach', 'TCGA-LIHC': 'Liver', 'TCGA-SKCM': 'Skin', 'TCGA-CESC': 'Cervix', 'TCGA-SARC': 'Soft Tissue', 'TCGA-ACC': 'Adrenal Gland/Paraganglia', 'TCGA-PCPG': 'Adrenal Gland/Paraganglia', 'TCGA-PAAD': 'Pancreas', 'TCGA-ESCA': 'Esophagus', 'TCGA-TGCT': 'Testis', 'TCGA-READ': 'Rectum', 'TCGA-THYM': 'Thymus', 'TCGA-UVM': 'Eye', 'TCGA-MESO': 'Pleura/Mesothelium', 'TCGA-OV': 'Ovary', 'TCGA-DLBC': 'Lymphatic System', 'TCGA-CHOL': 'Bile Duct'}
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger('extract_tcga_cna')

def load_barcode_to_case_id(path):
    df = pd.read_csv(path)

    return dict(zip(df['barcode'].astype(str), df['case_id'].astype(str)))

def load_barcode_to_tumor_origin(path):
    df = pd.read_csv(path)

    return {str(row['barcode']): TCGA_COHORT_TO_TUMOR_ORIGIN.get(row['cohort'], row['cohort']) for _, row in df.iterrows()}

def get_alteration_flags(gene_name, value):
    if pd.isna(value):
        return []
    try:
        value = int(round(float(value)))
    except (TypeError, ValueError):
        return []
    suffix = {-2: 'del', -1: 'loss', 1: 'gain', 2: 'amp'}.get(value)

    if suffix is None:
        return []
    return [f'cna_{gene_name}_any', f'cna_{gene_name}_{suffix}']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gistic-file', required=True)
    parser.add_argument('--case-id-barcode-map', default='/workspace/data/tcga/metadata/case_id_barcode_map.csv')
    parser.add_argument('--matched-cohort-csv', default='/workspace/data/tcga/metadata/matched_cohort.csv')
    parser.add_argument('--output-dir', default='/workspace/data/tcga/genomics/cna')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    barcode_to_case_id = load_barcode_to_case_id(Path(args.case_id_barcode_map))
    barcode_to_tumor_origin = load_barcode_to_tumor_origin(Path(args.matched_cohort_csv))
    gistic_df = pd.read_csv(args.gistic_file, sep='\t', index_col=0)
    records = []
    case_to_origin = {}
    unmatched = 0
    matched = 0

    for sample_col in tqdm(gistic_df.columns, desc='Processing samples', unit='sample'):
        barcode = str(sample_col)[:12]
        case_id = barcode_to_case_id.get(barcode)
        origin = barcode_to_tumor_origin.get(barcode)

        if case_id is None or origin is None:
            unmatched += 1
            continue
        matched += 1
        nonzero = gistic_df[sample_col][gistic_df[sample_col] != 0]

        for gene_name, value in nonzero.items():
            for flag in get_alteration_flags(str(gene_name), value):
                records.append((case_id, flag))
        case_to_origin[case_id] = origin
    if not records:
        raise RuntimeError('No CNA records were produced')
    records_df = pd.DataFrame(records, columns=['case_id', 'feature']).drop_duplicates()
    case_ids, row_idx = np.unique(records_df['case_id'].to_numpy(), return_inverse=True)
    columns, col_idx = np.unique(records_df['feature'].to_numpy(), return_inverse=True)
    matrix = coo_matrix((np.ones(len(records_df), dtype=np.int64), (row_idx, col_idx)), shape=(len(case_ids), len(columns))).tocsr()
    save_npz(output_dir / 'patient_cna_matrix.npz', matrix)
    (output_dir / 'patient_cna_matrix_rows.txt').write_text('\n'.join((f"{case_to_origin.get(cid, '')}\t{cid}" for cid in case_ids)))
    (output_dir / 'patient_cna_matrix_columns.txt').write_text('\n'.join(columns))
    counts = {suffix: sum((name.endswith('_' + suffix) for name in columns)) for suffix in ['any', 'del', 'loss', 'gain', 'amp']}
    summary = [f'GISTIC2 file samples: {gistic_df.shape[1]}', f'GISTIC2 file genes: {gistic_df.shape[0]}', f'Samples matched: {matched}', f'Samples with no match: {unmatched}', f'Final patient count: {len(case_ids)}', f'Final column count: {len(columns)}'] + [f'{key} features: {value}' for key, value in counts.items()]
    (output_dir / 'cna_extraction_summary.txt').write_text('\n'.join(summary))

if __name__ == '__main__':
    main()
