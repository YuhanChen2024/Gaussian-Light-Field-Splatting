import math
import torch
import torch.nn.functional as F


_WIN_CACHE = {}


def _to4d(img):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img.float()


# ──────────────────── PSNR ────────────────────

def calculate_psnr(img1, img2, data_range=1.0):
    """img1/img2: [0,1] 张量, 返回 float (dB)"""
    img1, img2 = _to4d(img1), _to4d(img2)
    mse = F.mse_loss(img1, img2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def rgb_to_y(img01):
    """RGB [0,1] → Y 亮度通道 [0,1] (BT.601)"""
    return (0.299 * img01[:, 0:1]
            + 0.587 * img01[:, 1:2]
            + 0.114 * img01[:, 2:3])


def calculate_psnr_y(img1, img2, data_range=1.0):
    return calculate_psnr(rgb_to_y(_to4d(img1)), rgb_to_y(_to4d(img2)),
                          data_range)


# ──────────────────── SSIM ────────────────────

def _gaussian_window(window_size, sigma, channel, device, dtype):
    key = (window_size, sigma, channel, str(device), str(dtype))
    if key not in _WIN_CACHE:
        coords = (torch.arange(window_size, dtype=dtype, device=device)
                  - window_size // 2)
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        w2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
        _WIN_CACHE[key] = w2d.expand(channel, 1, window_size,
                                     window_size).contiguous()
    return _WIN_CACHE[key]


def calculate_ssim(img1, img2, window_size=11, sigma=1.5, data_range=1.0):
    """标准 SSIM (高斯窗, 11x11, sigma=1.5), 返回 float"""
    img1, img2 = _to4d(img1), _to4d(img2)
    C = img1.size(1)
    win = _gaussian_window(window_size, sigma, C, img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, win, padding=pad, groups=C)
    mu2 = F.conv2d(img2, win, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, win, padding=pad,
                         groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, win, padding=pad,
                         groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, win, padding=pad,
                       groups=C) - mu1_mu2
    sigma1_sq = sigma1_sq.clamp(min=0)
    sigma2_sq = sigma2_sq.clamp(min=0)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
                / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))
    return ssim_map.mean().item()


# ──────────────── 亮度对齐 (无监督评估常用) ────────────────

def brightness_align(out01, gt01, eps=1e-6):
    """
    按 GT 全局均值对输出做线性亮度校正 (不改变结构/色相)。
    无监督增强方法在缺乏绝对曝光基准时的常规报告方式,
    Uses the ground-truth mean; distinct from raw-output metrics.
    """
    scale = gt01.mean() / out01.mean().clamp(min=eps)
    return (out01 * scale).clamp(0, 1)


# ──────────────── 一次性计算全部指标 ────────────────

def calculate_metrics(out01, gt01):
    """返回 dict: psnr / ssim / psnr_y / psnr_align / ssim_align"""
    out01, gt01 = _to4d(out01), _to4d(gt01)
    out_a = brightness_align(out01, gt01)
    return {
        'psnr': calculate_psnr(out01, gt01),
        'ssim': calculate_ssim(out01, gt01),
        'psnr_y': calculate_psnr_y(out01, gt01),
        'psnr_align': calculate_psnr(out_a, gt01),
        'ssim_align': calculate_ssim(out_a, gt01),
    }
