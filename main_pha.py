"""
main_v4.py — AdvancedHetDDI v3 + PharmaEnhancedHetDDI

PharmaEnhancedHetDDI: giữ nguyên KG + mol encoder (đã chứng minh tốt ở v3),
chỉ augment mol_emb bằng 19-dim global pharmacophore features qua learned gate.
Overhead nhỏ (~0.3M params), không thay đổi interface.

So sánh với v3 (baseline):
  v3: acc=0.675, f1=0.424, kappa=0.601
  v4: kỳ vọng acc~0.680+, f1~0.430+

Chạy PharmaEnhancedHetDDI:
    python main_v4.py --model PharmaEnhancedHetDDI --label_type multi_class \
        --condition s3 --mode concat --use_mixup --lr 5e-4

Chạy AdvancedHetDDI (baseline v3, để so sánh):
    python main_v4.py --model AdvancedHetDDI --label_type multi_class \
        --condition s3 --mode concat --use_mixup --lr 5e-4
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse

from model.AdvanceHetDDI import AdvancedHetDDI
from model.PharmacoDDI import PharmaEnhancedHetDDI
from utils.data_loader import load_data, get_train_test
from utils.pytorchtools import EarlyStopping
from utils.logger import Logger
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, cohen_kappa_score
)


# ---------------------------------------------------------------------------
# LR Scheduler: warmup → cosine decay
# ---------------------------------------------------------------------------
class WarmupCosine:
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 base_lr, min_ratio=0.01):
        self.opt = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.base_lr = base_lr
        self.min_lr = base_lr * min_ratio
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup:
            lr = self.base_lr * self._step / self.warmup
        else:
            progress = (self._step - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.opt.param_groups:
            pg['lr'] = lr


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def soft_ce(logits, soft_labels):
    """Cross-entropy with soft (mixed) labels."""
    return -(soft_labels * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def ce_smooth(logits, labels, smoothing, n_cls):
    """Label-smoothed cross-entropy."""
    oh = F.one_hot(labels, n_cls).float()
    smooth = oh * (1 - smoothing) + smoothing / n_cls
    return soft_ce(logits, smooth)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------
def resolve_device(arg):
    if arg == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        try:
            nn.Embedding(8, 8).to('cuda:0')(torch.tensor([0], device='cuda:0'))
            return torch.device('cuda:0')
        except RuntimeError:
            pass
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model(args, kg_g, smiles, class_num, device):
    common = dict(
        kg_g=kg_g, smiles=smiles,
        num_hidden=args.hidden_dim, num_layer=args.num_layer,
        mode=args.mode, class_num=class_num, condition=args.condition,
    )
    if args.model == 'PharmaEnhancedHetDDI':
        model = PharmaEnhancedHetDDI(**common).to(device)
    else:
        model = AdvancedHetDDI(**common).to(device)

    model.contrastive_weight = args.contrastive_weight
    model.label_smoothing    = args.label_smoothing
    return model


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------
def train_epoch(model, optimizer, scaler, use_amp,
                xl, xr, y_all,
                fold, epoch, batch_size, label_type, device, use_mixup):
    model.train()
    n = xl.shape[0]
    idx = np.random.permutation(n)
    total_loss = 0.0
    do_mixup = use_mixup and (epoch >= 20)
    n_cls     = model.class_num
    smoothing = getattr(model, 'label_smoothing',    0.05)
    c_weight  = getattr(model, 'contrastive_weight', 0.05)
    n_batches = 0

    for start in range(0, n, batch_size):
        bi = idx[start:start + batch_size]
        left  = torch.from_numpy(xl[bi]).long().to(device)
        right = torch.from_numpy(xr[bi]).long().to(device)
        y = y_all[bi].squeeze(-1).to(device) if label_type == 'multi_class' \
            else y_all[bi].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', enabled=use_amp):
            out = model(left, right,
                        labels=y,
                        return_contrastive=True,
                        use_mixup=do_mixup)
            if len(out) == 3:
                logits, c_loss, mixed_labels = out
                task_loss = soft_ce(logits, mixed_labels) \
                    if mixed_labels is not None \
                    else ce_smooth(logits, y, smoothing, n_cls)
            else:
                logits, c_loss = out
                task_loss = ce_smooth(logits, y, smoothing, n_cls) \
                    if label_type == 'multi_class' \
                    else nn.BCEWithLogitsLoss()(logits, y)
            loss = task_loss + c_weight * c_loss

        if not torch.isfinite(loss):
            print(f'[Fold {fold}][Epoch {epoch}] skip non-finite loss: {float(loss.detach().cpu())}')
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n_batches += 1

    print(f'[Fold {fold}][Epoch {epoch}] train loss: '
          f'{total_loss / max(1, n_batches):.4f}')


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_epoch(model, xl, xr, y_all,
               fold, epoch, batch_size, label_type, device):
    model.eval()
    n = xl.shape[0]
    all_logits, all_labels = [], []

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        left  = torch.from_numpy(xl[start:end]).long().to(device)
        right = torch.from_numpy(xr[start:end]).long().to(device)
        y = y_all[start:end].squeeze(-1).to(device) if label_type == 'multi_class' \
            else y_all[start:end].to(device)
        logits = model(left, right)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    preds     = all_logits.argmax(-1).numpy()
    labels_np = all_labels.numpy()
    acc   = accuracy_score(labels_np, preds)
    f1    = f1_score(labels_np, preds, average='macro', zero_division=0)
    prec  = precision_score(labels_np, preds, average='macro', zero_division=0)
    rec   = recall_score(labels_np, preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(labels_np, preds)

    print(f'[Fold {fold}][Epoch {epoch}] '
          f'acc:{acc:.3f} f1:{f1:.3f} '
          f'precision:{prec:.3f} recall:{rec:.3f} kappa:{kappa:.3f}')
    return [acc, f1, prec, rec, kappa]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run(args):
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    class_num = {
        'multi_class':  86,
        'binary_class': 1,
        'multi_label':  200,
    }[args.label_type]

    data_path = os.path.join(
        args.data_path, args.kg_name + '+' + args.ddi_name
    )
    kg_g, smiles = load_data(data_path, device=device)
    train_sample, test_sample = get_train_test(
        data_path,
        fold_num=args.fold_num,
        label_type=args.label_type,
        condition=args.condition,
    )

    # AMP can destabilize HGNN attention in this model family on some GPUs.
    # Keep it opt-in instead of always-on to avoid NaN at epoch 0.
    use_amp = (device.type == 'cuda' and args.use_amp)
    if use_amp:
        torch.backends.cudnn.benchmark = True

    all_scores = []

    for fold_i in range(args.fold_num):
        trl = train_sample[fold_i][:, 0]
        trr = train_sample[fold_i][:, 1]
        try_ = train_sample[fold_i][:, 2:]

        tel = test_sample[fold_i][:, 0]
        ter = test_sample[fold_i][:, 1]
        tey = test_sample[fold_i][:, 2:]

        if args.label_type == 'multi_class':
            try_ = torch.from_numpy(try_).long()
            tey  = torch.from_numpy(tey).long()
        else:
            try_ = torch.from_numpy(try_).float()
            tey  = torch.from_numpy(tey).float()

        model = build_model(args, kg_g, smiles, class_num, device)
        if fold_i == 0:
            print(model)

        # Parameter groups: no weight decay for bias/norm/embedding
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if any(k in name for k in ['bias', 'bn', 'norm', 'embedding']):
                bias_p.append(p)
            else:
                weight_p.append(p)

        optimizer = optim.AdamW(
            [
                {'params': weight_p, 'weight_decay': args.weight_decay},
                {'params': bias_p,   'weight_decay': 0.0},
            ],
            lr=args.lr,
        )
        scheduler     = WarmupCosine(optimizer, 10, args.epoch, args.lr)
        early_stop    = EarlyStopping(patience=args.patience, verbose=True)
        scaler        = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_score = None
        for epoch in range(args.epoch):
            train_epoch(
                model, optimizer, scaler, use_amp,
                trl, trr, try_,
                fold_i, epoch, args.batch_size,
                args.label_type, device, args.use_mixup,
            )
            scheduler.step()

            score = eval_epoch(
                model, tel, ter, tey,
                fold_i, epoch, args.batch_size,
                args.label_type, device,
            )

            # Only start early-stopping after warm-up phase
            if epoch > 50:
                early_stop(score[0], model)
                if early_stop.counter == 0:
                    best_score = score
                if early_stop.early_stop or epoch == args.epoch - 1:
                    break

            print(best_score)
            print("=" * 100)

        all_scores.append(best_score)
        print('Test set score:', all_scores)

    # Final summary
    avg = np.array(all_scores).mean(axis=0)
    print(
        f'\033[1;31mFinal DDI result:\n'
        f'acc:{avg[0]:.3f}, f1:{avg[1]:.3f}, '
        f'precision:{avg[2]:.3f}, recall:{avg[3]:.3f}, '
        f'kappa:{avg[4]:.3f}\033[0m'
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size',         type=int,   default=2**15)
    ap.add_argument('--fold_num',           type=int,   default=5)
    ap.add_argument('--hidden_dim',         type=int,   default=300)
    ap.add_argument('--num_layer',          type=int,   default=3)
    ap.add_argument('--epoch',              type=int,   default=1000)
    ap.add_argument('--patience',           type=int,   default=50)
    ap.add_argument('--lr',                 type=float, default=5e-4)
    ap.add_argument('--weight_decay',       type=float, default=1e-5)
    ap.add_argument('--label_type',         type=str,
                    choices=['multi_class', 'binary_class', 'multi_label'],
                    default='multi_class')
    ap.add_argument('--condition',          type=str,
                    choices=['s1', 's2', 's3'], default='s3')
    ap.add_argument('--mode',               type=str,
                    choices=['only_kg', 'only_mol', 'concat'],
                    default='concat')
    ap.add_argument('--data_path',          type=str,  default='./data')
    ap.add_argument('--kg_name',            type=str,  default='DRKG')
    ap.add_argument('--ddi_name',           type=str,
                    choices=['DrugBank', 'TWOSIDES'], default='DrugBank')
    ap.add_argument('--device',             type=str,
                    choices=['auto', 'cuda', 'cpu'], default='auto')
    ap.add_argument('--model',              type=str,
                    choices=['HetDDI', 'AdvancedHetDDI', 'PharmaEnhancedHetDDI'],
                    default='PharmaEnhancedHetDDI')
    ap.add_argument('--contrastive_weight', type=float, default=0.05)
    ap.add_argument('--label_smoothing',    type=float, default=0.05)
    ap.add_argument('--use_mixup',          action='store_true')
    ap.add_argument('--use_amp',            action='store_true')
    args = ap.parse_args()

    print(args)
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    device = resolve_device(args.device)
    print('running on', device)

    terminal = sys.stdout
    sys.stdout = Logger(
        f'./log/{args.model}_{args.label_type}_{args.condition}.txt',
        terminal,
    )

    run(args)