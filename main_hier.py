"""
main.py for HierDDI
"""
import sys, os, math, warnings, argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from model.HierDDI import HierDDI
from model.HetDDI import HetDDI
from model.AdvanceHetDDI import AdvancedHetDDI
from utils.data_loader import load_data, get_train_test
from train_test_hierddi import train_one_epoch, test, compute_class_weights
from utils.pytorchtools import EarlyStopping
from utils.logger import Logger


def resolve_device(device_arg):
    if device_arg == 'cpu': return torch.device('cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA not available.')
        return torch.device('cuda:0')
    if torch.cuda.is_available():
        try:
            nn.Embedding(8,8).to('cuda:0')(torch.tensor([0,1], device='cuda:0'))
            return torch.device('cuda:0')
        except RuntimeError as e:
            if 'no kernel image' in str(e).lower(): return torch.device('cpu')
            raise
    return torch.device('cpu')


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr_ratio=0.01):
        self.opt = optimizer
        self.warmup = warmup_epochs
        self.total  = total_epochs
        self.base   = base_lr
        self.min    = base_lr * min_lr_ratio
        self._ep    = 0

    def step(self):
        self._ep += 1
        if self._ep <= self.warmup:
            lr = self.base * self._ep / self.warmup
        else:
            p = (self._ep - self.warmup) / (self.total - self.warmup)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * p))
        for pg in self.opt.param_groups:
            pg['lr'] = lr


def build_model(args, kg_g, smiles, class_num, device):
    if args.model == 'HierDDI':
        model = HierDDI(
            kg_g, smiles,
            args.hidden_dim, args.num_layer,
            args.mode, class_num, args.condition,
            iga_rounds=args.iga_rounds,
        ).to(device)
        model.lambda_super    = args.lambda_super
        model.lambda_dis      = args.lambda_dis
        model.lambda_ncm      = args.lambda_ncm
        model.label_smoothing = args.label_smoothing

    elif args.model == 'AdvancedHetDDI':
        from model.AdvanceHetDDI import AdvancedHetDDI
        model = AdvancedHetDDI(
            kg_g, smiles,
            args.hidden_dim, args.num_layer,
            args.mode, class_num, args.condition,
        ).to(device)
        model.contrastive_weight = args.lambda_ncm
        model.label_smoothing    = args.label_smoothing

    else:
        model = HetDDI(
            kg_g, smiles,
            args.hidden_dim, args.num_layer,
            args.mode, class_num, args.condition,
        ).to(device)

    loss_func = nn.CrossEntropyLoss() if args.label_type == 'multi_class' else nn.BCEWithLogitsLoss()
    return model, loss_func


def run(args):
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)

    class_num = {'multi_class': 86, 'binary_class': 1, 'multi_label': 200}[args.label_type]
    data_path = os.path.join(args.data_path, args.kg_name + '+' + args.ddi_name)
    kg_g, smiles = load_data(data_path, device=device)
    train_sample, test_sample = get_train_test(
        data_path, fold_num=args.fold_num,
        label_type=args.label_type, condition=args.condition,
    )

    scores = []
    for i in range(args.fold_num):
        train_x_left  = train_sample[i][:, 0]
        train_x_right = train_sample[i][:, 1]
        train_y       = train_sample[i][:, 2:]
        test_x_left   = test_sample[i][:, 0]
        test_x_right  = test_sample[i][:, 1]
        test_y        = test_sample[i][:, 2:]

        if args.label_type == 'multi_class':
            train_y = torch.from_numpy(train_y).long()
            test_y  = torch.from_numpy(test_y).long()
        else:
            train_y = torch.from_numpy(train_y).float()
            test_y  = torch.from_numpy(test_y).float()

        model, loss_func = build_model(args, kg_g, smiles, class_num, device)
        if i == 0: print(model)

        # Class weights
        if args.label_type == 'multi_class' and args.use_class_weights:
            cw = compute_class_weights(train_y, class_num, device)
            model.class_weights = cw
            print(f'[Fold {i}] class_weights: min={cw.min():.2f} max={cw.max():.2f}')

        # Optimizer
        weight_p, other_p = [], []
        for name, p in model.named_parameters():
            if any(k in name for k in ('bias', 'bn', 'embedding', 'norm')):
                other_p.append(p)
            else:
                weight_p.append(p)

        optimizer = optim.Adam(
            [{'params': weight_p, 'weight_decay': args.weight_decay},
             {'params': other_p,  'weight_decay': 0}],
            lr=args.lr,
        )
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=10,
            total_epochs=args.epoch, base_lr=args.lr,
        )
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        best_score = None
        for epoch in range(args.epoch):
            train_one_epoch(
                model, loss_func, optimizer,
                train_x_left, train_x_right, train_y,
                i, epoch, args.batch_size, args.label_type, device,
                use_mixup=args.use_mixup,
            )
            scheduler.step()

            test_score = test(
                model, loss_func,
                test_x_left, test_x_right, test_y,
                i, epoch, args.batch_size, args.label_type, device,
            )

            if epoch > 50:
                early_stopping(test_score[0], model)
                if early_stopping.counter == 0:
                    best_score = test_score
                if early_stopping.early_stop or epoch == args.epoch - 1:
                    break

            print(best_score)
            print('=' * 100)

        scores.append(best_score)
        print('Test set score:', scores)

    scores = np.array(scores).mean(axis=0)
    if args.label_type == 'multi_class':
        print(
            "\033[1;31mFinal DDI result:\n"
            "acc:{:.3f}, f1:{:.3f}, precision:{:.3f}, recall:{:.3f}, kappa:{:.3f}\033[0m"
            .format(*scores[:5])
        )
    elif args.label_type == 'binary_class':
        print(
            "\033[1;31mFinal DDI result:\n"
            "acc:{:.3f}, f1:{:.3f}, precision:{:.3f}, recall:{:.3f}, auc:{:.3f}\033[0m"
            .format(*scores[:5])
        )
    else:
        print(scores)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size',   type=int,   default=2**15)
    ap.add_argument('--fold_num',     type=int,   default=5)
    ap.add_argument('--hidden_dim',   type=int,   default=300)
    ap.add_argument('--num_layer',    type=int,   default=3)
    ap.add_argument('--epoch',        type=int,   default=1000)
    ap.add_argument('--patience',     type=int,   default=50)
    ap.add_argument('--lr',           type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--label_type',   type=str,
                    choices=['multi_class','binary_class','multi_label'],
                    default='binary_class')
    ap.add_argument('--condition',    type=str,
                    choices=['s1','s2','s3'], default='s1')
    ap.add_argument('--mode',         type=str,
                    choices=['only_kg','only_mol','concat'], default='concat')
    ap.add_argument('--data_path',    type=str,  default='./data')
    ap.add_argument('--kg_name',      type=str,  default='DRKG')
    ap.add_argument('--ddi_name',     type=str,
                    choices=['DrugBank','TWOSIDES'], default='DrugBank')
    ap.add_argument('--device',       type=str,
                    choices=['auto','cuda','cpu'], default='auto')
    ap.add_argument('--model',        type=str,
                    choices=['HetDDI','AdvancedHetDDI','HierDDI'],
                    default='HierDDI')

    # HierDDI specific
    ap.add_argument('--iga_rounds',      type=int,   default=2,
                    help='Số round message passing trong IGA (khuyến nghị 2-3)')
    ap.add_argument('--lambda_super',    type=float, default=0.3,
                    help='Weight của super-class hierarchical loss')
    ap.add_argument('--lambda_dis',      type=float, default=0.1,
                    help='Weight của disentanglement loss')
    ap.add_argument('--lambda_ncm',      type=float, default=0.05,
                    help='Weight của NCM contrastive loss')
    ap.add_argument('--label_smoothing', type=float, default=0.05)
    ap.add_argument('--use_mixup',       action='store_true')
    ap.add_argument('--use_class_weights', action='store_true')

    args = ap.parse_args()
    print(args)

    terminal = sys.stdout
    log_file = './log/ddi_{}_{}_{}.txt'.format(
        args.model, args.label_type, args.condition
    )
    sys.stdout = Logger(log_file, terminal)
    warnings.filterwarnings('ignore', category=UserWarning)

    device = resolve_device(args.device)
    print('running on', device)
    run(args)