import os
import json
import math
import torch
import itertools
from collections import OrderedDict
from abc import ABC, abstractmethod
from utils import ImagePool
import networks

class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = (torch.device('cuda:{}'.format(self.gpu_ids[0]))
                       if self.gpu_ids else torch.device('cpu'))
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.preprocess != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.optimizers = []
        self.metric = 0

    @abstractmethod
    def set_input(self, inp):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self, opt):
        self.schedulers = [networks.get_scheduler(optimizer, opt)
                           for optimizer in self.optimizers]
        if opt.continue_train:
            load_suffix = ('iter_%d' % opt.load_iter
                           if opt.load_iter > 0 else opt.epoch)
            self.load_networks(load_suffix)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                getattr(self, 'net' + name).eval()

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                getattr(self, 'net' + name).train()

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

    def get_current_losses(self):
        return OrderedDict([(name, getattr(self, 'loss_' + name).item())
                            for name in self.loss_names
                            if isinstance(name, str)
                            and hasattr(self, 'loss_' + name)])

    def save_networks(self, epoch):
        """★ V4: 直接保存 state_dict 的 CPU 副本, 不搬动在线模型"""
        for name in self.model_names:
            if isinstance(name, str):
                save_path = os.path.join(
                    self.save_dir, '%s_net_%s.pth' % (epoch, name))
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                state = {k: v.detach().cpu()
                         for k, v in net.state_dict().items()}
                torch.save(state, save_path)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_path = os.path.join(
                    self.save_dir, '%s_net_%s.pth' % (epoch, name))
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path,
                                        map_location='cpu')
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def set_requires_grad(self, nets, requires_grad=False):
        for net in (nets if isinstance(nets, list) else [nets]):
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _get_net_module(self, net):
        return (net.module if isinstance(net, torch.nn.DataParallel)
                else net)

class CycleGANModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = [
            'D_A', 'G_A', 'cycle_A', 'idt_A', 'perceptual_A',
            'D_B', 'G_B', 'cycle_B', 'idt_B', 'perceptual_B',
            'expo_enhance', 'spatial_enhance', 'color_enhance',
            'color_angle',
            'tv_enhance', 'edge_enhance', 'edge_sobel',
            'ssim_cycle', 'illum_smooth', 'gs_smooth',
            'expo_dark',
        ]
        self.use_paired = bool(getattr(opt, 'paired_data', False))
        if self.use_paired:
            self.loss_names += ['paired_B', 'paired_A']

        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        n_blocks = getattr(opt, 'n_blocks', 4)
        num_heads = getattr(opt, 'num_heads', 4)
        use_ckpt = getattr(opt, 'use_checkpoint', False)

        self.netG_A = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
            opt.norm, not opt.no_dropout,
            opt.init_type, opt.init_gain, self.gpu_ids,
            n_blocks=n_blocks, num_heads=num_heads,
            use_checkpoint=use_ckpt)
        self.netG_B = networks.define_G(
            opt.output_nc, opt.input_nc, opt.ngf, opt.netG,
            opt.norm, not opt.no_dropout,
            opt.init_type, opt.init_gain, self.gpu_ids,
            n_blocks=n_blocks, num_heads=num_heads,
            use_checkpoint=use_ckpt)

        self.use_amp = getattr(opt, 'use_amp', False) and len(self.gpu_ids) > 0
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.best_metric_name = getattr(opt, 'best_metric', 'psnr')
        self.best_score = -float('inf')
        self.best_epoch = -1
        self.best_info_path = os.path.join(self.save_dir, 'best_info.json')

        self.netD_A = networks.define_D(
            opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
            opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_B = networks.define_D(
            opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D,
            opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.fake_A_pool = ImagePool(opt.pool_size)
        self.fake_B_pool = ImagePool(opt.pool_size)
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionL1 = torch.nn.L1Loss()

        self.criterionExposure = networks.ExposureControlLoss(
            patch_size=16, target_exposure=0.55)
        self.criterionSpatial = networks.SpatialConsistencyLoss().to(
            self.device)
        self.criterionColorConst = networks.ColorConstancyLoss()
        self.criterionTV = networks.TotalVariationLoss()
        self.criterionEdge = networks.EdgePreserveLoss()
        self.criterionSSIM = networks.SSIMLoss().to(self.device)
        self.criterionIllumSmooth = networks.IlluminationSmoothnessLoss()
        self.criterionGSSmooth = networks.GaussianSmoothnessLoss()
        self.criterionColorAngle = networks.ColorAngleLoss()
        self.criterionSobelEdge = networks.SobelEdgePreserveLoss().to(
            self.device)

        self.need_vgg = (opt.lambda_perceptual > 0
                         or (self.use_paired
                             and opt.lambda_paired_vgg > 0))
        if self.need_vgg:
            self.criterionVGG = networks.VGGLoss(
                self.device,
                n_layers=getattr(opt, 'vgg_layers', 3),
                max_size=getattr(opt, 'vgg_max_size', 0))
        else:
            self.criterionVGG = None

        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A.parameters(),
                            self.netG_B.parameters()),
            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(),
                            self.netD_B.parameters()),
            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers += [self.optimizer_G, self.optimizer_D]

    # ──────────────────────────────────────────
    def set_input(self, inp):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = inp['A' if AtoB else 'B'].to(self.device,
                                                   non_blocking=True)
        self.real_B = inp['B' if AtoB else 'A'].to(self.device,
                                                   non_blocking=True)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)   # G_A(normal) → dark
        self.rec_A = self.netG_B(self.fake_B)    # G_B(dark)   → normal (cycle)
        self.fake_A = self.netG_B(self.real_B)   # G_B(dark)   → normal ★增强★
        self.rec_B = self.netG_A(self.fake_A)    # G_A(normal) → dark   (cycle)
    def backward_D_basic(self, netD, real, fake):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pred_real = netD(real)
            pred_fake = netD(fake.detach())
            loss_D = (self.criterionGAN(pred_real, True) +
                      self.criterionGAN(pred_fake, False)) * 0.5
        if self.use_amp:
            self.scaler.scale(loss_D).backward()
        else:
            loss_D.backward()
        return loss_D

    def backward_D_A(self):
        self.loss_D_A = self.backward_D_basic(
            self.netD_A, self.real_B,
            self.fake_B_pool.query(self.fake_B))

    def backward_D_B(self):
        self.loss_D_B = self.backward_D_basic(
            self.netD_B, self.real_A,
            self.fake_A_pool.query(self.fake_A))

    def backward_G(self):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            lam = self.opt
            lambda_idt = lam.lambda_identity
            lambda_A = lam.lambda_A
            lambda_B = lam.lambda_B
            zero = torch.zeros((), device=self.device)


            self.loss_expo_enhance = (
                self.criterionExposure(self.fake_A) * lam.lambda_expo)
            self.loss_spatial_enhance = (
                self.criterionSpatial(self.fake_A, self.real_B)
                * lam.lambda_spatial)
            self.loss_color_enhance = (
                self.criterionColorConst(self.fake_A) * lam.lambda_color)
            self.loss_tv_enhance = (
                self.criterionTV(self.fake_A) * lam.lambda_tv)
            self.loss_edge_enhance = (
                self.criterionEdge(self.fake_A, self.real_B)
                * lam.lambda_edge)


            self.loss_color_angle = (
                self.criterionColorAngle(self.fake_A, self.real_B)
                * lam.lambda_color_angle)

            self.loss_edge_sobel = (
                (self.criterionSobelEdge(self.fake_A, self.real_B)
                 + 0.5 * self.criterionSobelEdge(self.rec_A, self.real_A)
                 + 0.5 * self.criterionSobelEdge(self.rec_B, self.real_B))
                * lam.lambda_edge_sobel)

            self.loss_ssim_cycle = (
                (self.criterionSSIM(self.rec_A, self.real_A)
                 + self.criterionSSIM(self.rec_B, self.real_B)) * 0.5
                * lam.lambda_ssim)

            illum_loss = zero
            for net_name in ['netG_A', 'netG_B']:
                net_mod = self._get_net_module(getattr(self, net_name))
                if (hasattr(net_mod, 'last_illum_field')
                        and net_mod.last_illum_field is not None):
                    illum_loss = illum_loss + self.criterionIllumSmooth(
                        net_mod.last_illum_field)
            self.loss_illum_smooth = illum_loss * lam.lambda_illum_smooth

            gs_loss = zero
            for net_name in ['netG_A', 'netG_B']:
                net_mod = self._get_net_module(getattr(self, net_name))
                if (hasattr(net_mod, 'last_alpha_refined')
                        and net_mod.last_alpha_refined is not None):
                    gs_loss = gs_loss + self.criterionGSSmooth(
                        net_mod.last_alpha_refined)
            self.loss_gs_smooth = gs_loss * lam.lambda_gs_smooth

            self.loss_expo_dark = (
                self.criterionExposure.forward_dark(self.fake_B)
                * lam.lambda_expo * 0.3)

            if lambda_idt > 0:
                self.idt_A = self.netG_A(self.real_B)
                self.loss_idt_A = (self.criterionIdt(self.idt_A, self.real_B)
                                   * lambda_B * lambda_idt)
                self.idt_B = self.netG_B(self.real_A)
                self.loss_idt_B = (self.criterionIdt(self.idt_B, self.real_A)
                                   * lambda_A * lambda_idt)
            else:
                self.idt_A = None
                self.idt_B = None
                self.loss_idt_A = zero
                self.loss_idt_B = zero

            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

            self.loss_cycle_A = (self.criterionCycle(self.rec_A, self.real_A)
                                 * lambda_A)
            self.loss_cycle_B = (self.criterionCycle(self.rec_B, self.real_B)
                                 * lambda_B)

            if lam.lambda_perceptual > 0 and self.criterionVGG is not None:
                self.loss_perceptual_A = (
                    self.criterionVGG(self.rec_B, self.real_B)
                    * lam.lambda_perceptual)
                self.loss_perceptual_B = (
                    self.criterionVGG(self.rec_A, self.real_A)
                    * lam.lambda_perceptual)
            else:
                self.loss_perceptual_A = zero
                self.loss_perceptual_B = zero

            if self.use_paired:
                paired_b = (self.criterionL1(self.fake_A, self.real_A)
                            * lam.lambda_paired
                            + self.criterionSSIM(self.fake_A, self.real_A)
                            * lam.lambda_paired_ssim)
                if lam.lambda_paired_vgg > 0 and self.criterionVGG is not None:
                    paired_b = paired_b + (
                        self.criterionVGG(self.fake_A, self.real_A)
                        * lam.lambda_paired_vgg)
                self.loss_paired_B = paired_b
                self.loss_paired_A = (
                    self.criterionL1(self.fake_B, self.real_B)
                    * lam.lambda_paired * 0.5)
            else:
                self.loss_paired_B = zero
                self.loss_paired_A = zero

            self.loss_G = (
                self.loss_G_A + self.loss_G_B
                + self.loss_cycle_A + self.loss_cycle_B
                + self.loss_idt_A + self.loss_idt_B
                + self.loss_perceptual_A + self.loss_perceptual_B
                + self.loss_expo_enhance + self.loss_spatial_enhance
                + self.loss_color_enhance + self.loss_color_angle
                + self.loss_tv_enhance
                + self.loss_edge_enhance + self.loss_edge_sobel
                + self.loss_ssim_cycle
                + self.loss_illum_smooth + self.loss_gs_smooth
                + self.loss_expo_dark
                + self.loss_paired_B + self.loss_paired_A
            )

        if self.use_amp:
            self.scaler.scale(self.loss_G).backward()
        else:
            self.loss_G.backward()

    # ──────────────────────────────────────────
    def optimize_parameters(self):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            self.forward()

        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad(set_to_none=True)
        self.backward_G()

        if self.use_amp:
            self.scaler.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(self.netG_A.parameters(),
                            self.netG_B.parameters()),
            max_norm=1.0)
        if self.use_amp:
            self.scaler.step(self.optimizer_G)
        else:
            self.optimizer_G.step()

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad(set_to_none=True)
        self.backward_D_A()
        self.backward_D_B()

        if self.use_amp:
            self.scaler.unscale_(self.optimizer_D)
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(self.netD_A.parameters(),
                            self.netD_B.parameters()),
            max_norm=1.0)
        if self.use_amp:
            self.scaler.step(self.optimizer_D)
            self.scaler.update()
        else:
            self.optimizer_D.step()

    # ──────────────────────────────────────────
    def get_quality_metric(self):
        """训练损失侧的辅助指标 (越低越好), 仅用于日志/无 GT 兜底"""
        metric = 0.0
        for name in ['cycle_A', 'cycle_B', 'ssim_cycle', 'edge_sobel']:
            if hasattr(self, f'loss_{name}'):
                val = getattr(self, f'loss_{name}')
                metric += (val.item() if isinstance(val, torch.Tensor)
                           else float(val))
        return metric

    def save_best_if_improved(self, score, epoch=None, extra=None):
        if score is None or not math.isfinite(score) or score <= self.best_score:
            return False
        self.best_score = float(score)
        self.best_epoch = epoch if epoch is not None else -1
        self.save_networks('best')
        self._dump_best_info(extra)
        return True

    def _dump_best_info(self, extra=None):
        info = {'metric': self.best_metric_name,
                'best_score': self.best_score,
                'best_epoch': self.best_epoch}
        if isinstance(extra, dict):
            info.update({k: float(v) for k, v in extra.items()})
        try:
            with open(self.best_info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f'[Warn] 无法写入 best_info.json: {e}')

    def load_best_info(self):
        if not os.path.isfile(self.best_info_path):
            return
        try:
            with open(self.best_info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            if info.get('metric') == self.best_metric_name:
                self.best_score = float(info.get('best_score', -float('inf')))
                self.best_epoch = int(info.get('best_epoch', -1))
                print(f'[Best] 已恢复历史最佳 {self.best_metric_name} = '
                      f'{self.best_score:.4f} (epoch {self.best_epoch})')
            else:
                print(f'[Best] 指标已从 {info.get("metric")} 切换为 '
                      f'{self.best_metric_name}, 重新开始记录')
        except (OSError, ValueError) as e:
            print(f'[Warn] 读取 best_info.json 失败: {e}')

def create_model(opt):
    return CycleGANModel(opt)
