import os
import random
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM',
                  '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                images.append(os.path.join(root, fname))
    return images[:min(max_dataset_size, len(images))]

def get_params(opt, size):
    """为配对样本生成共享的几何增强参数 (crop 位置 / 翻转)"""
    w, h = size
    new_h, new_w = h, w
    if 'resize' in opt.preprocess:
        new_h = new_w = opt.load_size
    elif 'scale_width' in opt.preprocess:
        new_w = opt.load_size
        new_h = max(opt.load_size * h // w, opt.crop_size)

    x = random.randint(0, max(0, new_w - opt.crop_size))
    y = random.randint(0, max(0, new_h - opt.crop_size))
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(
            lambda img: __make_power_2(img, base=8, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(
                lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    if ow > size or oh > size:
        return img.crop((x1, y1, x1 + size, y1 + size))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass

class UnalignedDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__(opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        if not self.A_size or not self.B_size:
            raise ValueError(f'No training images in {self.dir_A} or {self.dir_B}')
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1))

        # ★ 配对索引: 文件名(不含扩展名) → B 路径
        self.paired = getattr(opt, 'paired_data', False)
        self.B_by_stem = {}
        if self.paired:
            for p in self.B_paths:
                stem = os.path.splitext(os.path.basename(p))[0]
                self.B_by_stem.setdefault(stem, p)
            self.A_paths = [
                p for p in self.A_paths
                if os.path.splitext(os.path.basename(p))[0] in self.B_by_stem]
            self.A_size = len(self.A_paths)
            print(f'[Dataset] 配对监督已启用: {self.A_size} 对同名图像')
            assert self.A_size > 0, (
                'paired_data 已启用但 trainA/trainB 中没有同名文件, '
                '请检查数据集或去掉 --paired_data')

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        A_img = Image.open(A_path).convert('RGB')

        if not self.paired:
            index_B = (index % self.B_size
                       if self.opt.serial_batches
                       else random.randint(0, self.B_size - 1))
            B_path = self.B_paths[index_B]
            B_img = Image.open(B_path).convert('RGB')
            return {'A': self.transform_A(A_img), 'B': self.transform_B(B_img),
                    'A_paths': A_path, 'B_paths': B_path}

        # ── 配对模式: A 与 B 共享几何参数 ──
        stem = os.path.splitext(os.path.basename(A_path))[0]
        B_path = self.B_by_stem[stem]
        B_img = Image.open(B_path).convert('RGB')

        if A_img.size != B_img.size:
            raise ValueError(f'Paired images must have equal sizes: {A_path}, {B_path}')
        params = get_params(self.opt, A_img.size)
        tf_A = get_transform(self.opt, params,
                             grayscale=(self.input_nc == 1))
        tf_B = get_transform(self.opt, params,
                             grayscale=(self.output_nc == 1))
        return {'A': tf_A(A_img), 'B': tf_B(B_img),
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        if self.paired:
            return self.A_size
        return max(self.A_size, self.B_size)

class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = UnalignedDataset(opt)
        self.dataloader = data.DataLoader(
            self.dataset, batch_size=opt.batch_size,
            shuffle=not opt.serial_batches, num_workers=int(opt.num_threads),
            pin_memory=len(opt.gpu_ids) > 0, drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data_batch in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data_batch

def create_dataset(opt):
    return CustomDatasetDataLoader(opt).load_data()
