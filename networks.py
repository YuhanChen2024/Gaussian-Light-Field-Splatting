import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math
import torch.utils.checkpoint as cp

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True,
                                 track_running_stats=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False,
                                 track_running_stats=False)
    elif norm_type == 'none':
        return lambda x: Identity()
    raise NotImplementedError(f'normalization [{norm_type}] not found')

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def rule(epoch):
            return 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / \
                   float(opt.n_epochs_decay + 1)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)
    elif opt.lr_policy == 'step':
        return lr_scheduler.StepLR(optimizer,
                                   step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epochs, eta_min=0)

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_G(input_nc, output_nc, ngf, netG, norm='batch',
             use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[], n_blocks=4, num_heads=4, use_checkpoint=False):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf,
                              norm_layer=norm_layer,
                              use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf,
                              norm_layer=norm_layer,
                              use_dropout=use_dropout, n_blocks=6)
    elif netG == 'gs_vit':
        net = GSViTGenerator(input_nc, output_nc, ngf,
                             norm_layer=norm_layer,
                             n_blocks=n_blocks, num_heads=num_heads,
                             use_checkpoint=use_checkpoint)
    else:
        raise NotImplementedError(f'Generator [{netG}] not recognized')
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch',
             init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3,
                                  norm_layer=norm_layer)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D,
                                  norm_layer=norm_layer)
    elif netD == 'multi_scale':
        net = MultiScaleDiscriminator(input_nc, ndf, n_layers_D,
                                      norm_layer=norm_layer)
    else:
        raise NotImplementedError(f'Discriminator [{netD}] not recognized')
    return init_net(net, init_type, init_gain, gpu_ids)

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0,
                 target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None

    def _single(self, pred, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            t = self.real_label if target_is_real else self.fake_label
            return self.loss(pred, t.expand_as(pred))
        elif self.gan_mode == 'wgangp':
            return -pred.mean() if target_is_real else pred.mean()

    def __call__(self, prediction, target_is_real):
        if isinstance(prediction, list):
            return sum(self._single(p, target_is_real)
                       for p in prediction) / len(prediction)
        return self._single(prediction, target_is_real)

class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size=16, target_exposure=0.55):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target = target_exposure

    def forward(self, enhanced):
        img01 = (enhanced + 1.0) / 2.0
        local_mean = self.pool(img01)
        return (local_mean - self.target).pow(2).mean()

    def forward_dark(self, darkened):
        img01 = (darkened + 1.0) / 2.0
        local_mean = self.pool(img01)
        return (local_mean - 0.20).pow(2).mean()

class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('w_left',
            torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
            .unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_right',
            torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
            .unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_up',
            torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
            .unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_down',
            torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
            .unsqueeze(0).unsqueeze(0))

    def forward(self, enhanced, original):
        org_mean = torch.mean(original, dim=1, keepdim=True)
        enh_mean = torch.mean(enhanced, dim=1, keepdim=True)
        org_pool = F.avg_pool2d(org_mean, 4)
        enh_pool = F.avg_pool2d(enh_mean, 4)

        loss = 0
        for w in [self.w_left, self.w_right, self.w_up, self.w_down]:
            d_org = F.conv2d(org_pool, w, padding=1)
            d_enh = F.conv2d(enh_pool, w, padding=1)
            loss = loss + (d_org - d_enh).pow(2).mean()
        return loss

class ColorConstancyLoss(nn.Module):
    def forward(self, img):
        img01 = (img + 1.0) / 2.0
        mean_rgb = img01.mean(dim=[2, 3])
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        return ((mr - mg).pow(2) + (mr - mb).pow(2) + (mg - mb).pow(2)).mean()

class ColorAngleLoss(nn.Module):

    def forward(self, output, input_img):
        out01 = (output + 1.0) / 2.0
        in01 = (input_img + 1.0) / 2.0
        eps = 1e-7
        out_norm = F.normalize(out01 + eps, dim=1)
        in_norm = F.normalize(in01 + eps, dim=1)
        cos_sim = (out_norm * in_norm).sum(dim=1, keepdim=True)
        cos_sim = cos_sim.clamp(-1.0, 1.0)
        return (1.0 - cos_sim).mean()

class IlluminationSmoothnessLoss(nn.Module):
    def forward(self, illum_map):
        if illum_map is None:
            return torch.tensor(0.0)
        dh = (illum_map[:, :, 1:, :] - illum_map[:, :, :-1, :]).pow(2)
        dw = (illum_map[:, :, :, 1:] - illum_map[:, :, :, :-1]).pow(2)
        return dh.mean() + dw.mean()

class TotalVariationLoss(nn.Module):
    def forward(self, img):
        dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        return dh.mean() + dw.mean()

class EdgePreserveLoss(nn.Module):
    def forward(self, output, input_img):
        out_gx = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])
        out_gy = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])
        in_gx = torch.abs(input_img[:, :, :, 1:] - input_img[:, :, :, :-1])
        in_gy = torch.abs(input_img[:, :, 1:, :] - input_img[:, :, :-1, :])
        loss_x = F.relu(in_gx.detach() - out_gx).mean()
        loss_y = F.relu(in_gy.detach() - out_gy).mean()
        return loss_x + loss_y

class SobelEdgePreserveLoss(nn.Module):

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _rgb_to_y(self, img):
        img01 = (img + 1.0) / 2.0
        return (0.299 * img01[:, 0:1]
                + 0.587 * img01[:, 1:2]
                + 0.114 * img01[:, 2:3])

    def _sobel_edges(self, gray):
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return (gx ** 2 + gy ** 2 + 1e-8).sqrt()

    def forward(self, output, input_img):
        y_out = self._rgb_to_y(output)
        y_in = self._rgb_to_y(input_img)
        edge_out = self._sobel_edges(y_out)
        edge_in = self._sobel_edges(y_in)
        return F.l1_loss(edge_out, edge_in.detach())

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window',
                             self._create_window(window_size, channel))

    def _gaussian_1d(self, window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32
                               ) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size, channel):
        _1D = self._gaussian_1d(window_size, 1.5).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        C = img1.size(1)
        window = self.window
        if window.device != img1.device:
            window = window.to(img1.device)
        if C != self.channel:
            window = self._create_window(self.window_size, C).to(img1.device)
        pad = self.window_size // 2
        mu1 = F.conv2d(img1, window, padding=pad, groups=C)
        mu2 = F.conv2d(img2, window, padding=pad, groups=C)
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad,
                             groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad,
                             groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad,
                           groups=C) - mu1_mu2
        sigma1_sq = sigma1_sq.clamp(min=0)
        sigma2_sq = sigma2_sq.clamp(min=0)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()

class GaussianSmoothnessLoss(nn.Module):
    def forward(self, alpha):
        if alpha is None:
            return torch.tensor(0.0)
        dh = (alpha[:, :, 1:, :] - alpha[:, :, :-1, :]).abs()
        dw = (alpha[:, :, :, 1:] - alpha[:, :, :, :-1]).abs()
        return dh.mean() + dw.mean()

class VGGLoss(nn.Module):
    _RANGES = [(0, 4), (4, 9), (9, 16), (16, 23), (23, 30)]

    def __init__(self, device, loss_type="L1", n_layers=5, max_size=0):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        n_layers = max(1, min(int(n_layers), 5))
        self.slices = nn.ModuleList([
            nn.Sequential(*[vgg[x] for x in range(a, b)])
            for a, b in self._RANGES[:n_layers]]).to(device)
        self.max_size = int(max_size)
        for param in self.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss() if loss_type == "L1" else nn.MSELoss()

    def forward(self, x, y):
        if self.max_size > 0 and max(x.shape[-2:]) > self.max_size:
            size = (self.max_size, self.max_size)
            x = F.interpolate(x, size=size, mode='bilinear',
                              align_corners=False)
            y = F.interpolate(y, size=size, mode='bilinear',
                              align_corners=False)
        loss = 0
        for s in self.slices:
            x = s(x)
            y = s(y)
            loss = loss + self.criterion(x, y)
        return loss

def _enforce_pd_2d(sigma):
    a = sigma[:, 0:1].clamp(1.0, 200.0)
    c = sigma[:, 2:3].clamp(1.0, 200.0)
    b_raw = sigma[:, 1:2]
    b_max = (a * c).sqrt() * 0.5
    b = torch.clamp(b_raw, -b_max, b_max)
    return torch.cat([a, b, c], dim=1)

def _enforce_pd_seq(sigma):
    a = sigma[..., 0:1].clamp(1.0, 200.0)
    c = sigma[..., 2:3].clamp(1.0, 200.0)
    b_raw = sigma[..., 1:2]
    b_max = (a * c).sqrt() * 0.5
    b = torch.clamp(b_raw, -b_max, b_max)
    return torch.cat([a, b, c], dim=-1)

def _build_gaussian_kernel(ks=5, sigma=1.0):
    """构建 2D 高斯平滑核 (不可学习), 用于增益图平滑"""
    x = torch.arange(ks, dtype=torch.float32) - ks // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g2d = g.unsqueeze(1) * g.unsqueeze(0)
    g2d = g2d / g2d.sum()
    return g2d.view(1, 1, ks, ks)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask

class ChannelAttentionFusion(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.squeeze(x).view(B, C)
        w = self.excite(w).view(B, C, 1, 1)
        return x * w

class GaussianTokenizer(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.GELU()
        )
        self.geo = nn.Conv2d(in_dim, 6, 1)
        self.feat = nn.Conv2d(in_dim, feat_dim, 1)
        self.alpha_bias = nn.Parameter(torch.tensor(2.0))
        self.geo_smooth = nn.Conv2d(6, 6, 3, 1, 1, groups=6, bias=False)
        nn.init.constant_(self.geo_smooth.weight, 1.0 / 9.0)

    def forward(self, x):
        B, C, H, W = x.shape
        f = self.shared(x)
        geo_raw = self.geo(f)
        geo = self.geo_smooth(geo_raw)

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device), indexing='ij')
        base = torch.stack([xx, yy], 0).unsqueeze(0)

        delta = torch.tanh(geo[:, :2]) * (2.0 / min(H, W))
        mu = base + delta

        a = F.softplus(geo[:, 2:3]) + 1.0
        c = F.softplus(geo[:, 4:5]) + 1.0
        a, c = a.clamp(max=200.0), c.clamp(max=200.0)
        b_max = (a * c).sqrt() * 0.5
        b = torch.tanh(geo[:, 3:4]) * b_max
        sigma_inv = torch.cat([a, b, c], dim=1)

        alpha = torch.sigmoid(geo[:, 5:6] + self.alpha_bias)
        feat_out = self.feat(f)
        return mu, sigma_inv, alpha, feat_out

class MultiScaleGaussianTokenizer(nn.Module):
    def __init__(self, coarse_dim, fine_dim, feat_dim):
        super().__init__()
        self.coarse_tok = GaussianTokenizer(coarse_dim, feat_dim)
        self.fine_tok = GaussianTokenizer(fine_dim, feat_dim)

        self.fine_downsample = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, 2, 1),
            nn.GELU()
        )
        self.fuse_feat = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, 1),
            nn.GELU()
        )
        self.fuse_alpha = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, coarse_feat, fine_feat):
        mu_c, sig_c, alpha_c, feat_c = self.coarse_tok(coarse_feat)
        _, _, alpha_f, feat_f = self.fine_tok(fine_feat)

        feat_f_down = self.fine_downsample(feat_f)
        if feat_f_down.shape[2:] != feat_c.shape[2:]:
            feat_f_down = F.interpolate(feat_f_down, size=feat_c.shape[2:],
                                        mode='bilinear', align_corners=False)

        feat_fused = self.fuse_feat(torch.cat([feat_c, feat_f_down], dim=1))

        alpha_f_down = F.interpolate(alpha_f, size=alpha_c.shape[2:],
                                     mode='bilinear', align_corners=False)
        alpha_fused = self.fuse_alpha(
            torch.cat([alpha_c, alpha_f_down], dim=1))

        return mu_c, sig_c, alpha_fused, feat_fused

class GaussianIlluminationField(nn.Module):

    def __init__(self, feat_dim):
        super().__init__()
        self.illum_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(feat_dim // 4, 1, 1),
        )
        self.noise_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(feat_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        self.gain_refine = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Softplus()
        )
        self.gain_max = 4.0

    def forward(self, tokens_2d, alpha, target_h, target_w):
        illum_raw = self.illum_head(tokens_2d)
        illum_field = torch.sigmoid(illum_raw)

        noise_map = self.noise_head(tokens_2d)

        gain_input = torch.cat([illum_field, noise_map, alpha], dim=1)
        gain_low = self.gain_refine(gain_input) + 1.0
        gain_low = gain_low.clamp(max=self.gain_max)

        gain_full = F.interpolate(gain_low, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)

        return illum_field, gain_full

class GaussianSplattingAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.gs_logtemp = nn.Parameter(torch.zeros(num_heads))
        self.gs_blend = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, mu, sigma_inv):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale

        with torch.cuda.amp.autocast(enabled=False):
            mu_f = mu.float()
            sig_f = sigma_inv.float()
            dx = mu_f[:, :, 0].unsqueeze(2) - mu_f[:, :, 0].unsqueeze(1)
            dy = mu_f[:, :, 1].unsqueeze(2) - mu_f[:, :, 1].unsqueeze(1)
            a = sig_f[:, :, 0].unsqueeze(2)
            b_c = sig_f[:, :, 1].unsqueeze(2)
            c = sig_f[:, :, 2].unsqueeze(2)
            dist_sq = (a * dx ** 2 + 2 * b_c * dx * dy
                       + c * dy ** 2).clamp(0, 20)
            gs_affinity = torch.exp(-0.5 * dist_sq)
            temp = F.softplus(self.gs_logtemp).clamp(max=2.0)
            blend = torch.sigmoid(self.gs_blend)
            gs_scale = (blend * temp).view(1, -1, 1, 1)
            attn_f = attn_logits.float() + gs_affinity.unsqueeze(1) * gs_scale
            attn_f = attn_f.softmax(dim=-1)

        attn_out = attn_f.to(x.dtype)
        attn_out = self.attn_drop(attn_out)
        out = (attn_out @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))

class GSViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=3,
                 dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GaussianSplattingAttention(dim, num_heads, dropout)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim), nn.Dropout(dropout))
        self.drop_path2 = DropPath(drop_path)

        self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gs_refine = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.GELU(),
            nn.Linear(dim // 4, 6))
        self.gs_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens, mu, sigma_inv, alpha, H, W):
        B, N, D = tokens.shape

        t2d = tokens.permute(0, 2, 1).reshape(B, D, H, W)
        tokens = tokens + self.cpe(t2d).flatten(2).permute(0, 2, 1)

        tokens = tokens + self.drop_path1(
            self.attn(self.norm1(tokens), mu, sigma_inv))

        tokens = tokens + self.drop_path2(self.mlp(self.norm2(tokens)))

        gate = torch.tanh(self.gs_gate) * 0.3
        delta = self.gs_refine(tokens)
        mu = (mu + gate * torch.tanh(delta[..., :2]) * 0.01).clamp(-1, 1)
        sigma_raw = sigma_inv + gate * delta[..., 2:5] * 0.005
        sigma_inv = _enforce_pd_seq(sigma_raw)
        alpha = (alpha + gate * torch.tanh(delta[..., 5:6]) * 0.02
                 ).clamp(0, 1)
        return tokens, mu, sigma_inv, alpha

def _make_upsample_block(in_ch, out_ch, norm_layer):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        norm_layer(out_ch),
        nn.GELU()
    )

class GSViTGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=None,
                 n_blocks=4, num_heads=4, drop_path_rate=0.1,
                 use_checkpoint=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
                                           track_running_stats=False)
        feat_dim = ngf * 4

        self.use_checkpoint = use_checkpoint
        self.last_alpha_refined = None
        self.last_illum_field = None

        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7), norm_layer(ngf), nn.GELU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
            norm_layer(ngf * 2), nn.GELU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
            norm_layer(ngf * 4), nn.GELU())
        self.enc4 = nn.Sequential(
            nn.Conv2d(ngf * 4, feat_dim, 3, 2, 1),
            norm_layer(feat_dim), nn.GELU())

        self.ms_tokenizer = MultiScaleGaussianTokenizer(
            coarse_dim=feat_dim, fine_dim=ngf * 4, feat_dim=feat_dim)

        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, n_blocks)]
        self.blocks = nn.ModuleList([
            GSViTBlock(feat_dim, num_heads, mlp_ratio=3, dropout=0.1,
                       drop_path=dpr[i])
            for i in range(n_blocks)])
        self.out_norm = nn.LayerNorm(feat_dim)

        self.illum_field = GaussianIlluminationField(feat_dim)

        self.up1 = _make_upsample_block(feat_dim, feat_dim, norm_layer)
        self.ca3 = ChannelAttentionFusion(feat_dim)
        self.dec3 = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, 1, 1),
            norm_layer(feat_dim), nn.GELU())

        self.up2 = _make_upsample_block(feat_dim, ngf * 2, norm_layer)
        self.ca2 = ChannelAttentionFusion(ngf * 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1),
            norm_layer(ngf * 2), nn.GELU())

        self.up3 = _make_upsample_block(ngf * 2, ngf, norm_layer)
        self.ca1 = ChannelAttentionFusion(ngf)
        self.dec1 = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            norm_layer(ngf), nn.GELU())

        self.final_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7))

        self.skip_alpha = nn.Parameter(torch.tensor(-1.5))

        self.gain_inject = nn.Sequential(
            nn.Conv2d(ngf + 1, ngf, 3, 1, 1),
            nn.GELU())

        self.register_buffer('_gain_smooth_kernel',
                             _build_gaussian_kernel(ks=5, sigma=1.0))

    def forward(self, x):
        B, _, H_orig, W_orig = x.shape

        pad_h = (8 - H_orig % 8) % 8
        pad_w = (8 - W_orig % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x_input = x
        _, _, H, W = x.shape

        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)

        _, C, Hf, Wf = f4.shape

        mu_2d, sigma_2d, alpha_2d, gs_feat = \
            self.ms_tokenizer(f4, f3)

        tokens = gs_feat.flatten(2).permute(0, 2, 1)
        mu_seq = mu_2d.flatten(2).permute(0, 2, 1)
        sigma_seq = sigma_2d.flatten(2).permute(0, 2, 1)
        alpha_seq = alpha_2d.flatten(2).permute(0, 2, 1)

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                tokens, mu_seq, sigma_seq, alpha_seq = cp.checkpoint(
                    block, tokens, mu_seq, sigma_seq, alpha_seq, Hf, Wf,
                    use_reentrant=False)
            else:
                tokens, mu_seq, sigma_seq, alpha_seq = block(
                    tokens, mu_seq, sigma_seq, alpha_seq, Hf, Wf)
        tokens = self.out_norm(tokens)

        tokens_2d = tokens.permute(0, 2, 1).reshape(B, C, Hf, Wf)
        alpha_refined = alpha_seq.permute(0, 2, 1).reshape(B, 1, Hf, Wf)

        self.last_alpha_refined = alpha_refined

        illum_field_low, gain_full = \
            self.illum_field(tokens_2d, alpha_refined, H, W)

        gain_full = F.conv2d(gain_full, self._gain_smooth_kernel, padding=2)
        gain_full = gain_full.clamp(1.0, 4.0)

        self.last_illum_field = illum_field_low

        alpha_sm = F.avg_pool2d(alpha_refined, 5, 1, 2)
        alpha_soft = 0.9 + 0.1 * alpha_sm
        d = tokens_2d * alpha_soft

        d = self.up1(d)
        if d.shape[2:] != f3.shape[2:]:
            d = F.interpolate(d, size=f3.shape[2:],
                              mode='bilinear', align_corners=False)
        d = self.ca3(d + f3)
        d = self.dec3(d)

        d = self.up2(d)
        if d.shape[2:] != f2.shape[2:]:
            d = F.interpolate(d, size=f2.shape[2:],
                              mode='bilinear', align_corners=False)
        d = self.ca2(d + f2)
        d = self.dec2(d)

        d = self.up3(d)
        if d.shape[2:] != f1.shape[2:]:
            d = F.interpolate(d, size=f1.shape[2:],
                              mode='bilinear', align_corners=False)
        d = self.ca1(d + f1)
        d = self.dec1(d)

        gain_for_inject = F.interpolate(
            gain_full, size=d.shape[2:],
            mode='bilinear', align_corners=False)
        d = self.gain_inject(torch.cat([d, gain_for_inject], dim=1))

        residual = self.final_conv(d)
        out = torch.tanh(residual)

        skip_w = torch.sigmoid(self.skip_alpha)
        out = (1.0 - skip_w) * out + skip_w * x_input

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H_orig, :W_orig]

        return out

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, num_D=2):
        super().__init__()
        self.num_D = num_D
        self.discriminators = nn.ModuleList([
            NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
            for _ in range(num_D)])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1,
                                       count_include_pad=False)

    def forward(self, x):
        outputs = []
        for i, D in enumerate(self.discriminators):
            outputs.append(D(x))
            if i < self.num_D - 1:
                x = self.downsample(x)
        return outputs

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        seq = [nn.Conv2d(input_nc, ndf, 4, 2, 1),
               nn.LeakyReLU(0.2, True)]
        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4,
                              2, 1, bias=use_bias),
                    norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4,
                          1, 1, bias=use_bias),
                norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        conv_block += [nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias),
                       norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        conv_block += [nn.Conv2d(dim, dim, 3, padding=p, bias=use_bias),
                       norm_layer(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        super().__init__()
        self.last_alpha_refined = None
        self.last_illum_field = None
        use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, 7, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]
        for i in range(2):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, 2, 1,
                                bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 4, padding_type, norm_layer,
                                  use_dropout, use_bias)]
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [nn.ConvTranspose2d(ngf * mult,
                                         int(ngf * mult / 2),
                                         3, 2, 1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
