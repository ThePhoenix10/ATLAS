#!/usr/bin/env python3

import argparse
import gzip
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz
from tqdm import tqdm

TCGA_COHORT_TO_TUMOR_ORIGIN = {'TCGA-BRCA': 'Breast', 'TCGA-LUAD': 'Lung', 'TCGA-LUSC': 'Lung', 'TCGA-GBM': 'Brain', 'TCGA-LGG': 'Brain', 'TCGA-KICH': 'Kidney', 'TCGA-KIRC': 'Kidney', 'TCGA-KIRP': 'Kidney', 'TCGA-UCEC': 'Uterus', 'TCGA-UCS': 'Uterus', 'TCGA-HNSC': 'Head and Neck', 'TCGA-THCA': 'Thyroid', 'TCGA-COAD': 'Colon', 'TCGA-PRAD': 'Prostate', 'TCGA-BLCA': 'Bladder', 'TCGA-STAD': 'Stomach', 'TCGA-LIHC': 'Liver', 'TCGA-SKCM': 'Skin', 'TCGA-CESC': 'Cervix', 'TCGA-SARC': 'Soft Tissue', 'TCGA-ACC': 'Adrenal Gland/Paraganglia', 'TCGA-PCPG': 'Adrenal Gland/Paraganglia', 'TCGA-PAAD': 'Pancreas', 'TCGA-ESCA': 'Esophagus', 'TCGA-TGCT': 'Testis', 'TCGA-READ': 'Rectum', 'TCGA-THYM': 'Thymus', 'TCGA-UVM': 'Eye', 'TCGA-MESO': 'Pleura/Mesothelium', 'TCGA-OV': 'Ovary', 'TCGA-DLBC': 'Lymphatic System', 'TCGA-CHOL': 'Bile Duct'}
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger('extract_tcga_sbs96')
WANTED_COLUMNS = ['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2', 'Variant_Type', 'Tumor_Sample_Barcode', 'case_id']
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
PYRIMIDINES = {'C', 'T'}
SUBSTITUTION_TYPES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
BASES = ['A', 'C', 'G', 'T']
CHANNEL_NAMES = [f'{five}[{sub}]{three}' for sub in SUBSTITUTION_TYPES for five in BASES for three in BASES]

def get_channel(ref, alt, trinucleotide):
    five, _, three = trinucleotide

    if ref not in PYRIMIDINES:
        ref = COMPLEMENT[ref]
        alt = COMPLEMENT[alt]
        five, three = (COMPLEMENT[three], COMPLEMENT[five])
    return f'{five}[{ref}>{alt}]{three}'

def read_maf(path):
    with gzip.open(path, 'rt') as handle:
        header_line = 0

        for i, line in enumerate(handle):
            if not line.startswith('#'):
                header_line = i
                break
    df = pd.read_csv(path, sep='\t', skiprows=header_line, compression='gzip', low_memory=False)

    return df[[column for column in WANTED_COLUMNS if column in df.columns]]

def find_maf_files(maf_dir, file_ids):
    return [path for path in maf_dir.rglob('*.maf.gz') if path.parent.name in file_ids]

def build_patient_channels(df, genome):
    channels = set()
    snvs = df[df['Variant_Type'] == 'SNP'].dropna(subset=['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])

    for _, row in snvs.iterrows():
        chrom = str(row['Chromosome'])
        pos = int(row['Start_Position'])
        ref = str(row['Reference_Allele']).upper()
        alt = str(row['Tumor_Seq_Allele2']).upper()

        if ref not in 'ACGT' or alt not in 'ACGT' or ref == alt:
            continue
        try:
            trinucleotide = str(genome[chrom][pos - 2:pos + 1]).upper()
        except (KeyError, ValueError):
            continue
        if len(trinucleotide) != 3 or trinucleotide[1] != ref:
            continue
        channels.add(get_channel(ref, alt, trinucleotide))
    return channels

def load_barcode_to_origin(path):
    df = pd.read_csv(path)

    return {str(row['barcode']): TCGA_COHORT_TO_TUMOR_ORIGIN.get(row['cohort'], row['cohort']) for _, row in df.iterrows()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maf-dir', required=True)
    parser.add_argument('--matched-manifest', default='/workspace/data/tcga/metadata/matched_manifest.tsv')
    parser.add_argument('--matched-cohort-csv', default='/workspace/data/tcga/metadata/matched_cohort.csv')
    parser.add_argument('--reference-fasta', required=True)
    parser.add_argument('--output-dir', default='/workspace/data/tcga/genomics/sbs96')
    args = parser.parse_args()
    from pyfaidx import Fasta
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    genome = Fasta(args.reference_fasta)
    barcode_to_origin = load_barcode_to_origin(Path(args.matched_cohort_csv))
    manifest = pd.read_csv(args.matched_manifest, sep='\t')
    file_ids = set(manifest['id'].astype(str))
    maf_files = find_maf_files(Path(args.maf_dir), file_ids)
    records = []
    case_to_origin = {}
    failed = []
    zero_snv = 0

    for maf_path in tqdm(maf_files, desc='Computing SBS96 contexts', unit='file'):
        file_id = maf_path.parent.name

        try:
            df = read_maf(maf_path)

            if df.empty or 'case_id' not in df.columns:
                continue
            case_id = df['case_id'].iloc[0]
            channels = build_patient_channels(df, genome)

            if not channels:
                zero_snv += 1
                continue
            records.extend(((case_id, channel) for channel in channels))

            if 'Tumor_Sample_Barcode' in df.columns:
                barcode = str(df['Tumor_Sample_Barcode'].iloc[0])[:12]
                case_to_origin[case_id] = barcode_to_origin.get(barcode, '')
        except Exception as exc:
            failed.append((file_id, str(exc)))
    if not records:
        raise RuntimeError('No patients produced SBS96 features')
    records_df = pd.DataFrame(records, columns=['case_id', 'channel']).drop_duplicates()
    case_ids, row_idx = np.unique(records_df['case_id'].to_numpy(), return_inverse=True)
    channel_to_idx = {name: i for i, name in enumerate(CHANNEL_NAMES)}
    col_idx = records_df['channel'].map(channel_to_idx).to_numpy()
    matrix = coo_matrix((np.ones(len(records_df), dtype=np.int64), (row_idx, col_idx)), shape=(len(case_ids), len(CHANNEL_NAMES))).tocsr()
    save_npz(output_dir / 'patient_signature_matrix.npz', matrix)
    (output_dir / 'patient_signature_matrix_rows.txt').write_text('\n'.join((f"{case_to_origin.get(cid, '')}\t{cid}" for cid in case_ids)))
    (output_dir / 'patient_signature_matrix_columns.txt').write_text('\n'.join(CHANNEL_NAMES))
    summary = [f'MAF files found: {len(maf_files)}', f'Patients with SBS96 features: {len(case_ids)}', f'Patients skipped with zero usable SNVs: {zero_snv}', f'Files failed: {len(failed)}']

    if failed:
        summary += ['', 'Failed files:'] + [f'{file_id}: {reason}' for file_id, reason in failed[:50]]
    (output_dir / 'mutational_signatures_summary.txt').write_text('\n'.join(summary))

if __name__ == '__main__':
    main()
