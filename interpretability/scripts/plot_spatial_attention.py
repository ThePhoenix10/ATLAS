#!/usr/bin/env python3

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
CMAP = 'inferno'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--wsi', required=True)
    p.add_argument('--train_script', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--patient_id', default=None)
    p.add_argument('--true_label', default=None)
    p.add_argument('--render_level', type=int, default=2)
    p.add_argument('--patch_size_level0', type=float, default=512.0)
    p.add_argument('--dpi', type=int, default=600)

    return p.parse_args()

def import_model_class(train_script):
    import unittest.mock as mock
    spec = importlib.util.spec_from_file_location('train_module', train_script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['train_module'] = mod

    with mock.patch.object(mod, '__name__', 'train_module'):
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
        except Exception:
            pass
    if hasattr(mod, 'SpatialGATKAN'):
        print('[import] SpatialGATKAN found')

        return getattr(mod, 'SpatialGATKAN')
    raise RuntimeError('SpatialGATKAN not found in train script')

def load_model(ModelClass, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, nn.Module):
        return (ckpt.to(device).eval(), None)
    cfg = dict(input_dim=ckpt['input_dim'], hidden_dim=ckpt['hidden_dim'], num_classes=len(ckpt['classes']), genomic_input_dim=ckpt['genomic_input_dim'], genomic_intermediate_dim=ckpt.get('genomic_intermediate_dim', 512), num_layers=ckpt['num_layers'], num_heads=ckpt['num_heads'], kan_grid_size=ckpt['kan_grid_size'], radius_multiplier=ckpt['radius_multiplier'], kan_rank=ckpt['kan_rank'], kan_frequencies=ckpt['kan_frequencies'], dropout=ckpt['dropout'])
    print('[model] config:', cfg)
    model = ModelClass(**cfg)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(device).eval()
    print(f'[model] loaded — {sum((p.numel() for p in model.parameters())):,} params')

    return (model, ckpt)

def load_graph_features(graph):
    if 'x' in graph:
        return graph
    print("[graph] 'x' not in graph dict — loading from source_h5 ...")
    import h5py
    h5_path = str(graph['source_h5'])
    feat_key = str(graph.get('feature_key', 'features'))
    node_idx = graph['node_indices'].cpu().numpy().astype(int)

    with h5py.File(h5_path, 'r') as f:
        raw = f[feat_key][:]
        all_feats = raw[0] if raw.ndim == 3 else raw
        feats = all_feats[node_idx]
    graph['x'] = torch.tensor(feats, dtype=torch.float32)
    print(f"[graph] loaded features from h5: {tuple(graph['x'].shape)}")

    return graph

def get_positions(graph):
    if 'pos' in graph:
        return graph['pos'].cpu().numpy().astype(np.float32)
    import h5py
    h5_path = str(graph['source_h5'])
    node_idx = graph['node_indices'].cpu().numpy().astype(int)

    with h5py.File(h5_path, 'r') as f:
        for key in ('coords', 'coords_patching'):
            if key in f:
                raw = f[key][:]
                coords = raw[0] if raw.ndim == 3 else raw
                coords = coords[node_idx]
                print(f"[graph] loaded positions from H5 key '{key}': {coords.shape}")

                return coords.astype(np.float32)
    raise KeyError("No graph['pos'] and no coords/coords_patching dataset found in source_h5")

def extract_attention(model, graph, device):
    x = graph['x'].float().to(device)
    edge_index = graph['edge_index'].long().to(device)
    edge_attr = graph['edge_attr'].float().to(device)
    genomic_x = graph.get('genomic_x')

    if genomic_x is None:
        raise KeyError("ATLAS graph is missing 'genomic_x'")
    genomic_x = torch.as_tensor(genomic_x, dtype=torch.float32).reshape(1, -1).to(device)
    n = x.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)
    captured = {}
    pool = model.attention_pool
    orig_forward = pool.forward

    def patched_forward(x_, batch_, num_graphs_):
        pooled, weights = orig_forward(x_, batch_, num_graphs_)
        captured['weights'] = weights.detach()

        return (pooled, weights)
    pool.forward = patched_forward

    try:
        with torch.no_grad():
            _ = model(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, genomic_x=genomic_x)
    finally:
        pool.forward = orig_forward
    if 'weights' not in captured:
        raise RuntimeError('Could not capture attention_pool weights')
    w = captured['weights'].cpu().numpy().astype(np.float32).reshape(-1)

    if len(w) != n:
        raise RuntimeError(f'Attention length {len(w)} != node count {n}')
    print(f'[attn] extracted — min={w.min():.6f} max={w.max():.6f} sum={w.sum():.4f} nnz>1e-6={(w > 1e-06).sum()}')

    return w

def load_wsi_thumbnail(wsi_path, render_level):
    import openslide
    slide = openslide.OpenSlide(wsi_path)

    if render_level >= slide.level_count:
        raise ValueError(f'Requested render level {render_level}, but slide has only {slide.level_count} levels')
    level0_w, level0_h = slide.level_dimensions[0]
    render_w, render_h = slide.level_dimensions[render_level]
    region = slide.read_region((0, 0), render_level, slide.level_dimensions[render_level])
    thumb = np.array(region.convert('RGB'))
    slide.close()
    print(f'[wsi] level-0: {level0_w} x {level0_h}')
    print(f'[wsi] render level {render_level}: {render_w} x {render_h}')

    return (thumb, level0_w, level0_h, render_w, render_h)

def percentile_rank(arr):
    from scipy.stats import rankdata

    if len(arr) <= 1:
        return np.ones_like(arr, dtype=np.float32)
    return ((rankdata(arr) - 1) / (len(arr) - 1)).astype(np.float32)

def build_heatmap(x_rl, y_rl, attn_pct, canvas_w, canvas_h, tile_px):
    hmap = np.full((canvas_h, canvas_w), np.nan, dtype=np.float32)
    t = max(1, int(round(tile_px)))
    xi = np.clip(np.round(x_rl).astype(int), 0, canvas_w - 1)
    yi = np.clip(np.round(y_rl).astype(int), 0, canvas_h - 1)

    for idx in np.argsort(attn_pct):
        r0 = yi[idx]
        r1 = min(r0 + t, canvas_h)
        c0 = xi[idx]
        c1 = min(c0 + t, canvas_w)
        hmap[r0:r1, c0:c1] = attn_pct[idx]
    mask = ~np.isnan(hmap)
    hmap = np.nan_to_num(hmap, nan=0.0)

    return (hmap, mask)

def make_figure(graph, attn, thumb, positions, out_dir, dpi, patient_id, true_label, level0_w, level0_h, render_w, render_h, patch_size_level0):
    from scipy.ndimage import gaussian_filter
    level0_to_render_x = level0_w / render_w
    level0_to_render_y = level0_h / render_h
    x_l0 = positions[:, 0]
    y_l0 = positions[:, 1]
    x_rl = x_l0 / level0_to_render_x
    y_rl = y_l0 / level0_to_render_y
    tile_px = patch_size_level0 / level0_to_render_x
    edge_index = graph['edge_index'].cpu().numpy()
    attn_pct = percentile_rank(attn)
    cmap = plt.get_cmap(CMAP)
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    canvas_h = int(render_h) + 1
    canvas_w = int(render_w) + 1
    hmap, tissue_mask = build_heatmap(x_rl, y_rl, attn_pct, canvas_w, canvas_h, tile_px)
    hmap_smooth = gaussian_filter(hmap, sigma=max(tile_px * 3.0, 1.0))

    if tissue_mask.any():
        vals = hmap_smooth[tissue_mask]
        vmin = float(vals.min())
        vmax = float(vals.max())
        hmap_smooth = (hmap_smooth - vmin) / (vmax - vmin + 1e-12)
    rgba = cmap(hmap_smooth)
    rgba[..., 3] = np.where(tissue_mask, 0.6, 0.0)
    src_e, dst_e = (edge_index[0], edge_index[1])
    edge_score = attn_pct[src_e] * attn_pct[dst_e]

    if len(edge_score):
        edge_norm = (edge_score - edge_score.min()) / (edge_score.max() - edge_score.min() + 1e-12)
        keep = edge_norm >= np.percentile(edge_norm, 80)
        src_k = src_e[keep]
        dst_k = dst_e[keep]
        score_k = edge_norm[keep]
        print(f'[edges] keeping {keep.sum():,} / {len(keep):,} top-20% edges')
        segs_wsi = np.stack([np.stack([x_rl[src_k], y_rl[src_k]], axis=1), np.stack([x_rl[dst_k], y_rl[dst_k]], axis=1)], axis=1)
        edge_cmap = plt.get_cmap('inferno')
        all_colors = edge_cmap(score_k).copy()
        all_colors[:, 3] = 0.75 + score_k * 0.25
        all_lw = np.full(len(score_k), 0.2, dtype=np.float32)
    else:
        segs_wsi = np.empty((0, 2, 2))
        all_colors = np.empty((0, 4))
        all_lw = np.empty((0,))
    fig, (ax_hm, ax_edge) = plt.subplots(1, 2, figsize=(22, 9), facecolor='white', gridspec_kw={'wspace': 0.1})
    title = f'Patient: {patient_id}'

    if true_label:
        title += f'   |   Primary Origin: {true_label}'
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.87)
    ax_hm.imshow(thumb, origin='upper', interpolation='lanczos', extent=[0, render_w, render_h, 0], zorder=0)
    ax_hm.imshow(rgba, origin='upper', extent=[0, canvas_w, canvas_h, 0], interpolation='bilinear', zorder=1)

    if tissue_mask.any():
        ax_hm.contour(tissue_mask.astype(float), levels=[0.5], colors=['#333333'], linewidths=[0.4], origin='upper', extent=[0, canvas_w, canvas_h, 0], zorder=2)
    ax_hm.set_xlim(0, render_w)
    ax_hm.set_ylim(render_h, 0)
    ax_hm.set_aspect('equal')
    ax_hm.axis('off')
    ax_hm.set_title('A  —  Node Attention Heatmap', fontsize=13, loc='center', pad=10, fontweight='bold')
    cb1 = fig.colorbar(sm, ax=ax_hm, fraction=0.022, pad=0.01, shrink=0.72)
    cb1.set_label('Attention Score', fontsize=9, fontweight='bold', labelpad=10)
    thumb_blend = (thumb * 0.5).astype(np.uint8)
    ax_edge.imshow(thumb_blend, origin='upper', interpolation='lanczos', extent=[0, render_w, render_h, 0], zorder=0)

    if len(segs_wsi):
        lc_edge = LineCollection(segs_wsi, linewidths=all_lw, colors=all_colors, rasterized=True, zorder=1, capstyle='round')
        ax_edge.add_collection(lc_edge)
    ax_edge.set_xlim(0, render_w)
    ax_edge.set_ylim(render_h, 0)
    ax_edge.set_aspect('equal')
    ax_edge.axis('off')
    ax_edge.set_title('B  —  Edge Attention Heatmap', fontsize=13, loc='center', pad=10, fontweight='bold')
    sm_edge = ScalarMappable(cmap=plt.get_cmap('inferno'), norm=norm)
    sm_edge.set_array([])
    cb2 = fig.colorbar(sm_edge, ax=ax_edge, fraction=0.022, pad=0.01, shrink=0.72)
    cb2.set_label('Edge Attention Score', fontsize=9, fontweight='bold', labelpad=10)
    fig.text(0.5, 0.16, f'Nodes = {len(attn):,}     Edges = {edge_index.shape[1]:,}', ha='center', va='bottom', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.75', lw=0.8))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f'{patient_id}_atlas_spatial_attention'
    fig.savefig(str(stem) + '.pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(str(stem) + '.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[saved] {stem}.pdf')
    print(f'[saved] {stem}.png')

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    print('[graph] loading ...')
    graph = torch.load(args.graph, map_location='cpu', weights_only=False)
    graph = load_graph_features(graph)
    positions = get_positions(graph)
    print(f"[graph] {graph['x'].shape[0]} nodes, edge_index shape {tuple(graph['edge_index'].shape)}")

    if 'genomic_x' not in graph:
        raise KeyError('This is an ATLAS visualization and requires genomic_x in the graph.')
    print(f"[graph] genomic_x shape: {tuple(torch.as_tensor(graph['genomic_x']).reshape(-1).shape)}")
    ModelClass = import_model_class(args.train_script)
    model, ckpt = load_model(ModelClass, args.ckpt, device)
    print('[attn] running multimodal ATLAS forward pass ...')
    attn = extract_attention(model, graph, device)
    print('[wsi] reading thumbnail ...')
    thumb, level0_w, level0_h, render_w, render_h = load_wsi_thumbnail(args.wsi, args.render_level)
    patient_id = args.patient_id or str(graph.get('patient_id') or graph.get('case_id') or Path(args.graph).stem)
    true_label = args.true_label or graph.get('label_text') or ''
    print('[plot] rendering figure ...')
    make_figure(graph=graph, attn=attn, thumb=thumb, positions=positions, out_dir=Path(args.out_dir), dpi=args.dpi, patient_id=patient_id, true_label=true_label, level0_w=level0_w, level0_h=level0_h, render_w=render_w, render_h=render_h, patch_size_level0=args.patch_size_level0)
    print('[done]')

if __name__ == '__main__':
    main()
