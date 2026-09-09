#!/usr/bin/env python3

import argparse
import importlib.util
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

def load_training_module(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f'Training program not found: {path}')
    spec = importlib.util.spec_from_file_location('spatial_aware_training', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def normalize_label(x):
    return re.sub('[^a-z0-9]+', '', str(x).lower())

def resolve_true_label(raw_label, classes):
    if raw_label is None:
        return None
    raw = str(raw_label).strip()

    if not raw:
        return None
    if raw in classes:
        return raw
    norm_to_class = {normalize_label(c): c for c in classes}
    n = normalize_label(raw)

    if n in norm_to_class:
        return norm_to_class[n]
    aliases = {'breast': ['breast', 'brca'], 'colon': ['colon', 'coad', 'colorectal'], 'colorectal': ['colon', 'coad', 'colorectal'], 'kidney': ['kidney', 'renal', 'ccrcc', 'kirc'], 'lung': ['lung', 'luad', 'lusc', 'lscc'], 'ovary': ['ovary', 'ovarian', 'ov'], 'ovarian': ['ovary', 'ovarian', 'ov'], 'uterus': ['uterus', 'uterine', 'endometrial', 'ucec'], 'uterine': ['uterus', 'uterine', 'endometrial', 'ucec'], 'pancreas': ['pancreas', 'pancreatic', 'paad', 'pda'], 'pancreatic': ['pancreas', 'pancreatic', 'paad', 'pda'], 'brain': ['brain', 'gbm', 'glioblastoma'], 'headneck': ['headneck', 'headandneck', 'hnsc']}

    for key, terms in aliases.items():
        if n == normalize_label(key) or any((normalize_label(t) == n for t in terms)):
            hits = []

            for c in classes:
                cn = normalize_label(c)

                if any((normalize_label(t) in cn or cn in normalize_label(t) for t in terms)):
                    hits.append(c)
            hits = list(dict.fromkeys(hits))

            if len(hits) == 1:
                return hits[0]
    return None

def load_graph_item(graph_path, atlas):
    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    edge_index, edge_attr, node_indices = atlas.get_graph_components(graph)
    h5_path = atlas.resolve_h5_path(graph)
    x = atlas.load_feature_matrix(h5_path, node_indices=node_indices)
    x = torch.from_numpy(x).float()
    genomic_x = graph.get('genomic_x')

    if genomic_x is None:
        raise KeyError(f"'genomic_x' missing from {graph_path}")
    if not torch.is_tensor(genomic_x):
        genomic_x = torch.as_tensor(genomic_x, dtype=torch.float32)
    else:
        genomic_x = genomic_x.float()
    genomic_x = genomic_x.reshape(-1)
    patient_id = str(graph.get('patient_id') or graph.get('case_id') or Path(graph_path).stem)
    label_text = graph.get('label_text')

    return {'x': x, 'edge_index': edge_index.long(), 'edge_attr': edge_attr.float(), 'genomic_x': genomic_x, 'patient_id': patient_id, 'label_text': label_text}

def make_model(atlas, checkpoint, device):
    classes = checkpoint['classes']
    model = atlas.SpatialGATKAN(input_dim=int(checkpoint['input_dim']), hidden_dim=int(checkpoint['hidden_dim']), num_classes=len(classes), genomic_input_dim=int(checkpoint['genomic_input_dim']), genomic_intermediate_dim=int(checkpoint.get('genomic_intermediate_dim', 512)), num_layers=int(checkpoint['num_layers']), dropout=float(checkpoint['dropout']), num_heads=int(checkpoint['num_heads']), kan_grid_size=int(checkpoint['kan_grid_size']), radius_multiplier=float(checkpoint['radius_multiplier']), kan_rank=int(checkpoint['kan_rank']), kan_frequencies=int(checkpoint['kan_frequencies'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    return model

@torch.inference_mode()

def predict_one(model, item, device, use_amp):
    x = item['x'].to(device, non_blocking=True)
    edge_index = item['edge_index'].to(device, non_blocking=True)
    edge_attr = item['edge_attr'].to(device, non_blocking=True)
    genomic_x = item['genomic_x'].unsqueeze(0).to(device, non_blocking=True)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        logits = model(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, genomic_x=genomic_x)
    probs = F.softmax(logits.float(), dim=1)[0].cpu().numpy()
    del x, edge_index, edge_attr, genomic_x, batch, logits

    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer-script', default='/workspace/scripts/train_spatial_aware.py', help='Spatial-aware training source.')
    parser.add_argument('--results-dir', default='/workspace/results/spatial-aware', help='Five-fold spatial-aware results containing fold_i/best_model.pt.')
    parser.add_argument('--graph-dir', default='/workspace/data/cptac/spatial-aware')
    parser.add_argument('--output-dir', default='/workspace/results/cptac')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--only-fold', type=int, default=None)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    graph_dir = Path(args.graph_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_paths = sorted(graph_dir.glob('*.pt'))

    if not graph_paths:
        raise RuntimeError(f'No CPTAC .pt graphs found in {graph_dir}')
    print(f'CPTAC graphs found: {len(graph_paths):,}')
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    use_amp = device.type == 'cuda'
    print(f'Device: {device}')
    print('Loading exact original ATLAS architecture...')
    atlas = load_training_module(args.trainer_script)
    folds = [args.only_fold] if args.only_fold is not None else list(range(1, args.n_folds + 1))
    checkpoints = []
    reference_classes = None

    for fold in folds:
        ckpt_path = results_dir / f'fold_{fold}' / 'best_model.pt'

        if not ckpt_path.exists():
            raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        classes = list(ckpt['classes'])

        if reference_classes is None:
            reference_classes = classes
        elif classes != reference_classes:
            raise RuntimeError(f'Class order differs in fold {fold}; cannot safely ensemble.')
        checkpoints.append((fold, ckpt_path))
    classes = reference_classes
    print(f'TCGA output classes: {len(classes)}')
    print('Classes:', classes)
    print(f'Folds to evaluate: {folds}')
    patient_info = []

    for p in tqdm(graph_paths, desc='Reading CPTAC graph metadata'):
        graph = torch.load(p, map_location='cpu', weights_only=False)
        pid = str(graph.get('patient_id') or graph.get('case_id') or p.stem)
        raw_label = graph.get('label_text')
        patient_info.append({'graph_path': p, 'patient_id': pid, 'raw_label': raw_label})
        del graph
    prob_sum = np.zeros((len(graph_paths), len(classes)), dtype=np.float64)
    fold_prediction_columns = {}

    for fold, ckpt_path in checkpoints:
        print()
        print('=' * 100)
        print(f'FOLD {fold}: {ckpt_path}')
        print('=' * 100)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = make_model(atlas, checkpoint, device)
        fold_preds = []

        for i, info in enumerate(tqdm(patient_info, desc=f'Fold {fold} CPTAC inference')):
            item = load_graph_item(info['graph_path'], atlas)

            if item['x'].shape[1] != int(checkpoint['input_dim']):
                raise RuntimeError(f"{info['patient_id']}: pathology dim {item['x'].shape[1]} != checkpoint {checkpoint['input_dim']}")
            if item['genomic_x'].numel() != int(checkpoint['genomic_input_dim']):
                raise RuntimeError(f"{info['patient_id']}: genomic dim {item['genomic_x'].numel()} != checkpoint {checkpoint['genomic_input_dim']}")
            probs = predict_one(model=model, item=item, device=device, use_amp=use_amp)
            prob_sum[i] += probs
            fold_preds.append(classes[int(np.argmax(probs))])
            del item
        fold_prediction_columns[f'fold_{fold}_prediction'] = fold_preds
        del model, checkpoint

        if device.type == 'cuda':
            torch.cuda.empty_cache()
    ensemble_probs = prob_sum / len(checkpoints)
    top1_idx = np.argmax(ensemble_probs, axis=1)
    top3_idx = np.argsort(-ensemble_probs, axis=1)[:, :3]
    rows = []

    for i, info in enumerate(patient_info):
        pred = classes[int(top1_idx[i])]
        top3 = [classes[int(j)] for j in top3_idx[i]]
        mapped_true = resolve_true_label(info['raw_label'], classes)
        row = {'patient_id': info['patient_id'], 'raw_true_label': info['raw_label'], 'mapped_true_label': mapped_true, 'ensemble_prediction': pred, 'ensemble_top1_probability': float(ensemble_probs[i, top1_idx[i]]), 'ensemble_top3': ' | '.join(top3), 'top3_correct': bool(mapped_true in top3) if mapped_true is not None else None}

        for col, vals in fold_prediction_columns.items():
            row[col] = vals[i]
        for class_idx, class_name in enumerate(classes):
            row[f'prob_{class_name}'] = float(ensemble_probs[i, class_idx])
        rows.append(row)
    pred_df = pd.DataFrame(rows)
    pred_path = output_dir / 'cptac_predictions.csv'
    pred_df.to_csv(pred_path, index=False)
    mapped = pred_df['mapped_true_label'].notna()
    scored = pred_df.loc[mapped].copy()
    metrics = {'n_graphs': int(len(pred_df)), 'n_scored': int(len(scored)), 'n_unmapped_labels': int((~mapped).sum()), 'folds_ensembled': folds, 'checkpoint_results_dir': str(results_dir), 'graph_dir': str(graph_dir)}

    if len(scored) > 0:
        y_true = scored['mapped_true_label'].tolist()
        y_pred = scored['ensemble_prediction'].tolist()
        top1 = accuracy_score(y_true, y_pred)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=classes, zero_division=0)
        top3 = float(scored['top3_correct'].astype(bool).mean())
        metrics.update({'top1_accuracy': float(top1), 'top3_accuracy': float(top3), 'weighted_f1': float(weighted_f1)})
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(output_dir / 'cptac_confusion_matrix.csv')
        report = classification_report(y_true, y_pred, labels=classes, target_names=classes, output_dict=True, zero_division=0)
        (output_dir / 'cptac_classification_report.json').write_text(json.dumps(report, indent=2))
    unmapped_labels = sorted({str(x) for x in pred_df.loc[pred_df['mapped_true_label'].isna(), 'raw_true_label'].dropna().tolist()})
    metrics['unmapped_raw_labels'] = unmapped_labels
    metrics_path = output_dir / 'cptac_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print()
    print('=' * 100)
    print('CPTAC EXTERNAL VALIDATION — ORIGINAL 93.46% GATED ATLAS')
    print('=' * 100)
    print(f'Patients/graphs evaluated: {len(pred_df):,}')
    print(f'Patients with mapped labels: {len(scored):,}')
    print(f'Unmapped labels: {(~mapped).sum():,}')

    if 'top1_accuracy' in metrics:
        print(f"Top-1:       {metrics['top1_accuracy'] * 100:.2f}%")
        print(f"Top-3:       {metrics['top3_accuracy'] * 100:.2f}%")
        print(f"Weighted F1: {metrics['weighted_f1'] * 100:.2f}%")
    else:
        print('No labels could be mapped, so prediction files were saved without scoring.')
    if unmapped_labels:
        print('Unmapped raw label values:')

        for x in unmapped_labels:
            print('  ', x)
    print('=' * 100)
    print(f'Predictions: {pred_path}')
    print(f'Metrics:     {metrics_path}')
    print(f'Output dir:  {output_dir}')

if __name__ == '__main__':
    main()
