import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True,
                            help='path to images (should have trainA/trainB)')
        parser.add_argument('--name', type=str, default='experiment_name')
        parser.add_argument('--gpu_ids', type=str, default='0')
        parser.add_argument('--checkpoints_dir', type=str,
                            default='./checkpoints')
        parser.add_argument('--input_nc', type=int, default=3)
        parser.add_argument('--output_nc', type=int, default=3)
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--ndf', type=int, default=64)
        parser.add_argument('--netD', type=str, default='multi_scale')
        parser.add_argument('--netG', type=str, default='gs_vit',
                            help='gs_vit | resnet_9blocks | resnet_6blocks')
        parser.add_argument('--n_layers_D', type=int, default=3)
        parser.add_argument('--norm', type=str, default='instance')
        parser.add_argument('--init_type', type=str, default='normal')
        parser.add_argument('--init_gain', type=float, default=0.02)
        parser.add_argument('--no_dropout', action='store_true', default=True)
        parser.add_argument('--direction', type=str, default='AtoB',
                            help='AtoB: A=normal, B=lowlight')
        parser.add_argument('--serial_batches', action='store_true')
        parser.add_argument('--num_threads', default=4, type=int)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--load_size', type=int, default=286)
        parser.add_argument('--crop_size', type=int, default=256)
        parser.add_argument('--max_dataset_size', type=int,
                            default=float("inf"))
        parser.add_argument('--preprocess', type=str,
                            default='resize_and_crop')
        parser.add_argument('--no_flip', action='store_true')
        parser.add_argument('--epoch', type=str, default='latest',
                            help='latest | best | <epoch>')
        parser.add_argument('--load_iter', type=int, default=0)

        parser.add_argument('--use_amp', action='store_true',
                            help='启用 AMP 混合精度训练 (省显存, 建议开)')
        parser.add_argument('--use_checkpoint', action='store_true',
                            help='GS-ViT block 梯度检查点 — 显存换算力, '
                                 '省约 40%% 峰值, 速度慢 ~25%%')
        parser.add_argument('--vgg_layers', type=int, default=3,
                            help='VGG 感知损失使用的层数 (1~5), 越少越省显存')
        parser.add_argument('--vgg_max_size', type=int, default=0,
                            help='VGG 前先降采样到该分辨率, 0=不降采样 '
                                 '(显存紧张可设 128)')

        # 生成器结构参数
        parser.add_argument('--n_blocks', type=int, default=4,
                            help='GS-ViT Transformer block 数量')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='GS-ViT 注意力头数')

        parser.add_argument('--eval_dir', type=str, default='./eval',
                            help='评估根目录, 需包含 low/ 与 high/ 子目录 '
                                 '(支持再嵌套一层)')
        parser.add_argument('--eval_low_dir', type=str, default='',
                            help='弱光图目录 (留空则自动取 eval_dir/low)')
        parser.add_argument('--eval_high_dir', type=str, default='',
                            help='参考图目录 (留空则自动取 eval_dir/high)')
        parser.add_argument('--eval_freq', type=int, default=1,
                            help='每 N 个 epoch 评估一次 (默认每轮)')
        parser.add_argument('--eval_size', type=int, default=256,
                            help='评估时统一缩放到该分辨率')
        parser.add_argument('--eval_full_res', action='store_true',
                            help='按原始分辨率评估 (显存占用高, 可能 OOM)')
        parser.add_argument('--best_metric', type=str, default='psnr',
                            choices=['psnr', 'ssim', 'psnr_y',
                                     'psnr_align', 'combined'],
                            help='best checkpoint 选择依据 '
                                 '(combined = psnr + 10*ssim)')

        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        opt = parser.parse_args()

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = [int(i) for i in str_ids if int(i) >= 0]
        if opt.gpu_ids:
            if not torch.cuda.is_available():
                parser.error('CUDA unavailable; use --gpu_ids -1 for CPU')
            torch.cuda.set_device(opt.gpu_ids[0])
        for key in ('batch_size', 'print_freq', 'save_latest_freq', 'eval_freq'):
            if getattr(opt, key) <= 0:
                parser.error(f'--{key} must be positive')
        if opt.num_threads < 0 or opt.eval_size <= 0:
            parser.error('num_threads must be >= 0 and eval_size > 0')
        if opt.input_nc != 3 or opt.output_nc != 3:
            parser.error('GLFS training and evaluation require RGB (3 channels)')

        self.opt = opt
        return self.opt

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--save_latest_freq', type=int, default=5000)
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--epoch_count', type=int, default=1)
        parser.add_argument('--phase', type=str, default='train')

        parser.add_argument('--n_epochs', type=int, default=50)
        parser.add_argument('--n_epochs_decay', type=int, default=50)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--gan_mode', type=str, default='lsgan')
        parser.add_argument('--pool_size', type=int, default=50)
        parser.add_argument('--lr_policy', type=str, default='linear')
        parser.add_argument('--lr_decay_iters', type=int, default=50)

        parser.add_argument('--lambda_A', type=float, default=10.0,
                            help='cycle consistency A')
        parser.add_argument('--lambda_B', type=float, default=10.0,
                            help='cycle consistency B')
        parser.add_argument('--lambda_identity', type=float, default=0.5,
                            help='identity loss (设 0 可省 2 次前向显存)')
        parser.add_argument('--lambda_perceptual', type=float, default=5.0,
                            help='循环重建感知损失 (VGG), 设 0 则跳过计算')

        parser.add_argument('--lambda_expo', type=float, default=1.0,
                            help='曝光控制损失')
        parser.add_argument('--lambda_spatial', type=float, default=1.0,
                            help='空间一致性损失')
        parser.add_argument('--lambda_color', type=float, default=0.1,
                            help='色彩恒常性损失')
        parser.add_argument('--lambda_color_angle', type=float, default=1.0,
                            help='★ 色彩向量夹角损失 — 严格保色相')
        parser.add_argument('--lambda_tv', type=float, default=0.005,
                            help='总变分正则')
        parser.add_argument('--lambda_edge', type=float, default=0.2,
                            help='边缘保持损失')
        parser.add_argument('--lambda_edge_sobel', type=float, default=1.0,
                            help='★ Sobel 边缘损失 — 拉高 SSIM')
        parser.add_argument('--lambda_ssim', type=float, default=3.0,
                            help='SSIM 结构相似性 (cycle)')
        parser.add_argument('--lambda_illum_smooth', type=float, default=1.0,
                            help='光照场平滑正则')
        parser.add_argument('--lambda_gs_smooth', type=float, default=0.5,
                            help='GS alpha 平滑正则')

        parser.add_argument('--paired_data', action='store_true',
                            help='trainA/trainB 同名配对时启用配对监督; '
                                 'B 直接取 A 的配对图并共享几何增强, '
                                 '不增加任何前向次数')
        parser.add_argument('--lambda_paired', type=float, default=10.0,
                            help='配对 L1 监督权重')
        parser.add_argument('--lambda_paired_ssim', type=float, default=3.0,
                            help='配对 SSIM 监督权重')
        parser.add_argument('--lambda_paired_vgg', type=float, default=1.0,
                            help='配对 VGG 感知监督权重 (0 则跳过计算)')

        return parser
