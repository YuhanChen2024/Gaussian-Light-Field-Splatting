import argparse
import contextlib
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import networks
from metrics import calculate_metrics, calculate_ssim, rgb_to_y

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

METRIC_KEYS = ['psnr', 'ssim', 'psnr_y', 'ssim_y', 'mae',
               'psnr_align', 'ssim_align']

# ════════════════════════ 参数 ════════════════════════

def get_args():
    p = argparse.ArgumentParser('GLFS 原分辨率验证脚本')

    # ── 数据 ──
    p.add_argument('--eval_dir', type=str, default='./eval',
                   help='评估根目录, 需含 low/ 与 high/ (支持再嵌套一层)')
    p.add_argument('--low_dir', type=str, default='',
                   help='弱光目录 (留空 = eval_dir/low)')
    p.add_argument('--high_dir', type=str, default='',
                   help='参考目录 (留空 = eval_dir/high; 留空且不存在则只推理)')

    # ── 权重 ──
    p.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    p.add_argument('--name', required=True, help='Checkpoint subdirectory name')
    p.add_argument('--epoch', type=str, default='best',
                   help='best | latest | <epoch 数字>')
    p.add_argument('--direction', type=str, default='BtoA',
                   choices=['AtoB', 'BtoA'],
                   help='BtoA=增强(G_B), AtoB=压暗(G_A)')

    # ── 网络结构 (须与训练一致) ──
    p.add_argument('--netG', type=str, default='gs_vit')
    p.add_argument('--input_nc', type=int, default=3)
    p.add_argument('--output_nc', type=int, default=3)
    p.add_argument('--ngf', type=int, default=64)
    p.add_argument('--norm', type=str, default='instance')
    p.add_argument('--n_blocks', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=4)

    # ── 运行 ──
    p.add_argument('--gpu_id', type=int, default=0, help='-1 = CPU')
    p.add_argument('--amp', action='store_true',
                   help='用 autocast 半精度推理 (省显存, 指标略有波动)')
    p.add_argument('--max_side', type=int, default=0,
                   help='>0 时长边超过该值才等比缩小 (默认 0 = 严格原分辨率)')
    p.add_argument('--metric_on_cpu', action='store_true',
                   help='指标在 CPU 上算 (超大图省显存)')

    # ── 分块回退 ──
    p.add_argument('--tile', type=int, default=512,
                   help='分块推理的块边长 (仅 OOM 回退或 --force_tile 时使用)')
    p.add_argument('--tile_overlap', type=int, default=96)
    p.add_argument('--force_tile', action='store_true',
                   help='强制分块推理 (全局光照估计会受块划分影响)')
    p.add_argument('--no_auto_tile', action='store_true',
                   help='OOM 时不回退分块, 直接报错')

    # ── 输出 ──
    p.add_argument('--output', type=str, default='./eval_out/fullres')
    p.add_argument('--lpips', action='store_true',
                   help='计算 LPIPS (需 pip install lpips)')
    args = p.parse_args()
    if args.tile < 8 or not 0 <= args.tile_overlap < (args.tile // 8 * 8):
        p.error('Require tile >= 8 and 0 <= tile_overlap < floor(tile/8)*8')
    if args.max_side < 0:
        p.error('--max_side must be >= 0')
    if args.input_nc != 3 or args.output_nc != 3:
        p.error('Evaluation requires RGB (3 channels)')
    return args

# ════════════════════════ 数据 ════════════════════════

def collect_recursive(d):
    out = []
    if not d or not os.path.isdir(d):
        return out
    for root, _, fnames in os.walk(d):
        for fn in fnames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTENSIONS:
                out.append(os.path.join(root, fn))
    return sorted(out)

def resolve_dirs(args):
    low = args.low_dir or (os.path.join(args.eval_dir, 'low')
                           if args.eval_dir else '')
    high = args.high_dir or (os.path.join(args.eval_dir, 'high')
                             if args.eval_dir else '')
    if not os.path.isdir(low):
        raise FileNotFoundError(f'弱光目录不存在: {low}')
    if args.high_dir and not os.path.isdir(high):
        raise FileNotFoundError(f'Reference directory not found: {high}')
    if not os.path.isdir(high):
        print(f'[Warn] 参考目录不存在 ({high}), 将只做推理不算指标')
        high = ''
    return low, high

def build_pairs(low_dir, high_dir):
    lows = collect_recursive(low_dir)
    if not lows:
        raise FileNotFoundError(f'{low_dir} 中未找到图片')
    if not high_dir:
        return [(p, None) for p in lows]

    high_by_stem = {}
    for p in collect_recursive(high_dir):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in high_by_stem:
            raise ValueError(f'Duplicate GT stem: {stem}')
        high_by_stem[stem] = p

    pairs, missing = [], []
    for lp in lows:
        stem = os.path.splitext(os.path.basename(lp))[0]
        if stem in high_by_stem:
            pairs.append((lp, high_by_stem[stem]))
        else:
            missing.append(stem)
    if missing:
        print(f'[Warn] {len(missing)} 张弱光图无同名参考, 已跳过指标: '
              f'{missing[:8]}{" ..." if len(missing) > 8 else ""}')
    if not pairs:
        raise RuntimeError('low/high 没有任何同名配对, 请检查目录结构')
    return pairs

def load01(path, max_side=0, size=None):
    """读图 → [0,1] 的 (1,3,H,W); max_side>0 时才等比缩小"""
    img = Image.open(path).convert('RGB')
    if size is not None and size > 0:
        img = img.resize((size, size), Image.BICUBIC)
    elif max_side > 0 and max(img.size) > max_side:
        w, h = img.size
        s = max_side / float(max(w, h))
        img = img.resize((max(1, int(round(w * s))),
                          max(1, int(round(h * s)))), Image.BICUBIC)
    return transforms.functional.to_tensor(img).unsqueeze(0)

# ════════════════════════ 模型 ════════════════════════

def load_generator(args, device):
    """构建裸生成器 (不包 DataParallel) 并加载权重"""
    if args.direction == 'AtoB':          # G_A: normal → dark
        in_nc, out_nc, tag = args.input_nc, args.output_nc, 'G_A'
    else:                                  # G_B: dark → normal (增强)
        in_nc, out_nc, tag = args.output_nc, args.input_nc, 'G_B'

    net = networks.define_G(
        in_nc, out_nc, args.ngf, args.netG, args.norm,
        use_dropout=False, init_type='normal', init_gain=0.02,
        gpu_ids=[],                        # 关键: 不做 DataParallel 包装
        n_blocks=args.n_blocks, num_heads=args.num_heads,
        use_checkpoint=False)

    save_dir = os.path.join(args.checkpoints_dir, args.name)
    cand = [f'{args.epoch}_net_{tag}.pth', f'{args.epoch}_net_G.pth']
    load_path = next((os.path.join(save_dir, c) for c in cand
                      if os.path.isfile(os.path.join(save_dir, c))), None)
    if load_path is None:
        avail = sorted(f for f in os.listdir(save_dir)
                       if f.endswith('.pth')) if os.path.isdir(save_dir) else []
        raise FileNotFoundError(
            f'找不到权重 {cand} @ {save_dir}\n可用: {avail}')

    state = torch.load(load_path, map_location='cpu')
    if hasattr(state, '_metadata'):
        del state._metadata
    # 兼容带 module. 前缀的旧权重
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)

    net.to(device).eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    print(f'[Model] 已加载 {load_path}')
    print(f'[Model] 参数量: {sum(p.numel() for p in net.parameters())/1e6:.2f} M')
    return net, load_path

# ════════════════════════ 推理 ════════════════════════

def _amp_ctx(enabled, device):
    if enabled and device.type == 'cuda':
        return torch.cuda.amp.autocast(enabled=True)
    return contextlib.nullcontext()

def _blend_window(h, w, device, dtype):
    wy = torch.hann_window(h, periodic=False,
                           device=device, dtype=dtype).clamp_min(1e-2)
    wx = torch.hann_window(w, periodic=False,
                           device=device, dtype=dtype).clamp_min(1e-2)
    return (wy[:, None] * wx[None, :]).view(1, 1, h, w)

def _tile_starts(total, tile, stride):
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts

@torch.no_grad()
def forward_tiled(net, x, tile, overlap, amp, device):
    """Overlap-tiled inference with Hann-window blending."""
    B, C, H, W = x.shape
    tile = tile // 8 * 8
    th, tw = min(tile, H), min(tile, W)
    sh, sw = max(1, th - overlap), max(1, tw - overlap)

    acc = None
    wsum = torch.zeros(1, 1, H, W, device=x.device, dtype=torch.float32)
    win = _blend_window(th, tw, x.device, torch.float32)

    for y in _tile_starts(H, th, sh):
        for xx in _tile_starts(W, tw, sw):
            patch = x[:, :, y:y + th, xx:xx + tw]
            with _amp_ctx(amp, device):
                o = net(patch).float()
            if acc is None:
                acc = torch.zeros(B, o.shape[1], H, W, device=x.device, dtype=torch.float32)
            acc[:, :, y:y + th, xx:xx + tw] += o * win
            wsum[:, :, y:y + th, xx:xx + tw] += win
    return acc / wsum.clamp_min(1e-6)

@torch.no_grad()
def enhance(net, low01, args, device):
    """low01: [0,1] → (out01, mode)"""
    x = (low01 * 2.0 - 1.0).to(device, non_blocking=True)

    if args.force_tile:
        out = forward_tiled(net, x, args.tile, args.tile_overlap,
                            args.amp, device)
        return ((out + 1.0) / 2.0).clamp(0, 1), 'tiled'

    try:
        with _amp_ctx(args.amp, device):
            out = net(x).float()
        return ((out + 1.0) / 2.0).clamp(0, 1), 'full'
    except RuntimeError as e:
        if args.no_auto_tile or 'out of memory' not in str(e).lower():
            raise
    print(f'[OOM] Retrying with tiles ({args.tile}, overlap={args.tile_overlap})')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    out = forward_tiled(net, x, args.tile, args.tile_overlap, args.amp, device)
    return ((out + 1.0) / 2.0).clamp(0, 1), 'tiled'

# ════════════════════════ 指标 ════════════════════════

def compute_metrics(out01, gt01, lpips_fn=None):
    """out01/gt01: 同尺寸 [0,1] 4D 张量"""
    res = calculate_metrics(out01, gt01)
    res['ssim_y'] = calculate_ssim(rgb_to_y(out01), rgb_to_y(gt01))
    res['mae'] = (out01 - gt01).abs().mean().item()
    if lpips_fn is not None:
        with torch.no_grad():
            res['lpips'] = lpips_fn(out01 * 2 - 1, gt01 * 2 - 1).mean().item()
    return res

def summarize(rows, keys):
    stats = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None],
                     dtype=np.float64)
        if v.size == 0:
            continue
        stats[k] = {'mean': float(v.mean()), 'std': float(v.std()),
                    'min': float(v.min()), 'max': float(v.max())}
    return stats

# ════════════════════════ 主流程 ════════════════════════

def main():
    args = get_args()
    if args.gpu_id >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --gpu_id -1 for CPU")
    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    low_dir, high_dir = resolve_dirs(args)
    pairs = build_pairs(low_dir, high_dir)
    net, ckpt_path = load_generator(args, device)
    metric_device = torch.device('cpu') if args.metric_on_cpu else device
    lpips_fn = None
    if args.lpips and high_dir:
        try:
            import lpips as lpips_lib
        except ImportError as exc:
            raise ImportError('Install lpips: pip install lpips') from exc
        lpips_fn = lpips_lib.LPIPS(net='alex').to(metric_device).eval()

    enh_dir = os.path.join(args.output, 'enhanced')
    os.makedirs(enh_dir, exist_ok=True)
    rows, total_t = [], 0.0

    for idx, (low_path, high_path) in enumerate(pairs, 1):
        stem = os.path.splitext(os.path.basename(low_path))[0]
        low01 = load01(low_path, args.max_side)
        H, W = low01.shape[-2:]
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out01, mode = enhance(net, low01, args, device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        total_t += dt

        # Preserve nested input paths to prevent overwriting equal file stems.
        rel = os.path.relpath(low_path, low_dir)
        out_path = os.path.join(enh_dir, os.path.splitext(rel)[0] + '.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_image(out01, out_path)
        row = {'filename': stem, 'relative_path': rel, 'height': H, 'width': W,
               'mode': mode, 'time_s': round(dt, 4)}

        if high_path is not None:
            gt01 = load01(high_path, args.max_side).to(metric_device)
            pred = out01.to(metric_device)
            if gt01.shape[-2:] != pred.shape[-2:]:
                print(f'[Warn] {stem}: resizing output to GT for metrics only')
                pred = F.interpolate(pred, size=gt01.shape[-2:], mode='bicubic',
                                     align_corners=False).clamp(0, 1)
            row.update(compute_metrics(pred, gt01, lpips_fn))
        rows.append(row)
        msg = f'[{idx}/{len(pairs)}] {stem} {W}x{H} {mode} {dt:.3f}s'
        if 'psnr' in row:
            msg += f' PSNR={row["psnr"]:.3f} SSIM={row["ssim"]:.4f}'
        print(msg)

    keys = [k for k in METRIC_KEYS + ['lpips'] if any(k in r for r in rows)]
    fields = ['filename', 'relative_path', 'height', 'width', 'mode', 'time_s'] + keys
    with open(os.path.join(args.output, 'per_image_metrics.csv'), 'w',
              newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'checkpoint': ckpt_path, 'direction': args.direction,
        'num_images': len(rows), 'num_paired': sum('psnr' in r for r in rows),
        'max_side': args.max_side, 'amp': bool(args.amp and device.type == 'cuda'),
        'tiled_images': sum(r['mode'] == 'tiled' for r in rows),
        'avg_time_s': total_t / len(rows),
        'peak_gpu_mem_GB': (torch.cuda.max_memory_allocated(device) / 1024**3
                            if device.type == 'cuda' else 0.0),
        'metrics': summarize(rows, keys),
    }
    with open(os.path.join(args.output, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if 'psnr_align' in keys:
        print('Note: *_align metrics use the GT mean and are not raw-output metrics.')
    print(f'Enhanced images: {enh_dir}')

if __name__ == '__main__':
    main()
