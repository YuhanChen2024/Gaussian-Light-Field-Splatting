import csv
import json
import os
import time
import torch
import torch.nn.functional as F
from dataset import create_dataset
from evaluate import build_pairs, load01
from metrics import calculate_metrics
from models import create_model
from options import TrainOptions


METRIC_KEYS = ('psnr', 'ssim', 'psnr_y', 'psnr_align', 'ssim_align')


def validation_pairs(opt):
    low = opt.eval_low_dir or os.path.join(opt.eval_dir, 'low')
    high = opt.eval_high_dir or os.path.join(opt.eval_dir, 'high')
    if opt.eval_low_dir and not os.path.isdir(low):
        raise FileNotFoundError(low)
    if opt.eval_high_dir and not os.path.isdir(high):
        raise FileNotFoundError(high)
    if not (opt.eval_dir or opt.eval_low_dir or opt.eval_high_dir):
        return []
    if not os.path.isdir(low) or not os.path.isdir(high):
        print('[Validation] No paired validation directory; best uses training loss.')
        return []
    return build_pairs(low, high)


@torch.no_grad()
def evaluate_with_reference(model, pairs, opt):
    net = model.netG_B
    was_training = net.training
    net.eval()
    size = None if opt.eval_full_res else opt.eval_size
    acc = {key: 0.0 for key in METRIC_KEYS}
    try:
        for low_path, high_path in pairs:
            low = load01(low_path, size=size).to(model.device)
            gt = load01(high_path, size=size).to(model.device)
            out = ((net(low * 2 - 1) + 1) / 2).clamp(0, 1)
            if out.shape[-2:] != gt.shape[-2:]:
                out = F.interpolate(out, size=gt.shape[-2:], mode='bicubic',
                                    align_corners=False).clamp(0, 1)
            for key, value in calculate_metrics(out, gt).items():
                acc[key] += value
        return {key: value / len(pairs) for key, value in acc.items()}
    finally:
        net.train(was_training)


def pick_score(metrics, name):
    if name == 'combined':
        return metrics['psnr'] + 10 * metrics['ssim']
    return metrics[name]


def main():
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    if len(dataset.dataloader) == 0:
        raise ValueError('No training batches; decrease --batch_size or add images.')
    model = create_model(opt)
    model.setup(opt)
    pairs = validation_pairs(opt)
    model.best_metric_name = opt.best_metric if pairs else 'negative_train_loss'
    if opt.continue_train:
        model.load_best_info()
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    with open(os.path.join(expr_dir, 'train_config.json'), 'w', encoding='utf-8') as f:
        config = vars(opt).copy()
        if config['max_dataset_size'] == float('inf'):
            config['max_dataset_size'] = None
        json.dump(config, f, indent=2, ensure_ascii=False)
    log_path = os.path.join(expr_dir, 'eval_log.csv')
    fields = ['epoch', *METRIC_KEYS, 'train_loss_metric', 'score',
              'best_score', 'best_epoch', 'is_best', 'epoch_time_s']
    if not os.path.isfile(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    print(f'Train images: {len(dataset)} | Validation pairs: {len(pairs)}')
    print(f'Best metric: {model.best_metric_name} | AMP: {model.use_amp}')

    total_iters = 0
    last_epoch = opt.n_epochs + opt.n_epochs_decay
    for epoch in range(opt.epoch_count, last_epoch + 1):
        start = time.perf_counter()
        model.train()
        quality_sum, n_batches = 0.0, 0
        for batch in dataset:
            total_iters += opt.batch_size
            model.set_input(batch)
            model.optimize_parameters()
            quality_sum += model.get_quality_metric()
            n_batches += 1

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                print(f'Epoch[{epoch}/{last_epoch}] Iter[{total_iters}] ' +
                      ', '.join(f'{k}:{v:.4f}' for k, v in losses.items()))
            if total_iters % opt.save_latest_freq == 0:
                model.save_networks('latest')

        if n_batches == 0:
            raise RuntimeError('No training batches produced.')
        avg_loss = quality_sum / n_batches
        model.update_learning_rate()
        model.save_networks('latest')

        metrics, score, improved = {}, None, False
        if pairs and epoch % opt.eval_freq == 0:
            metrics = evaluate_with_reference(model, pairs, opt)
            score = pick_score(metrics, opt.best_metric)
            if opt.lr_policy == 'plateau':
                model.metric = -score
            improved = model.save_best_if_improved(score, epoch, metrics)
        elif not pairs:
            score = -avg_loss
            improved = model.save_best_if_improved(score, epoch)

        elapsed = time.perf_counter() - start
        row = {'epoch': epoch, **metrics, 'train_loss_metric': avg_loss,
               'score': score, 'best_score': model.best_score,
               'best_epoch': model.best_epoch, 'is_best': int(improved),
               'epoch_time_s': elapsed}
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        metric_text = ' '.join(f'{k}={v:.4f}' for k, v in metrics.items())
        print(f'Epoch {epoch}: loss={avg_loss:.4f} {metric_text} '
              f'best={model.best_score:.4f} @ {model.best_epoch} '
              f'{"[new best] " if improved else ""}time={elapsed:.1f}s')

    print(f'Checkpoints: {expr_dir}')


if __name__ == '__main__':
    main()
