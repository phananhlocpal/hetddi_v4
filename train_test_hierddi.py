"""
train_test.py cho HierDDI
Xử lý hierarchical loss: CE_leaf + λ_super * CE_super + λ_dis * DisLoss + λ_ncm * NCMLoss
"""
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, cohen_kappa_score, roc_auc_score,
)
from model.HierDDI import HierDDI


def _is_hier(model) -> bool:
    return isinstance(model, HierDDI)


def soft_cross_entropy(logits, soft_labels, class_weights=None):
    log_prob = F.log_softmax(logits, dim=-1)
    if class_weights is not None:
        log_prob = log_prob * class_weights.unsqueeze(0)
    return -(soft_labels * log_prob).sum(dim=-1).mean()


def label_smooth(labels, num_classes, smoothing, device):
    oh = F.one_hot(labels, num_classes).float().to(device)
    return oh * (1 - smoothing) + smoothing / num_classes


def compute_class_weights(train_y, num_classes, device):
    labels = train_y.squeeze(-1).long()
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()
    counts = counts.clamp(min=1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    weights = weights.clamp(max=5.0)
    return weights.to(device)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, loss_func, optimizer,
    train_x_left, train_x_right, train_y,
    fold, epoch, batch_size, label_type, device,
    use_mixup=False,
):
    model.train()
    total_loss = 0.0
    n = train_x_left.shape[0]
    idx = np.random.permutation(n)

    smoothing  = getattr(model, 'label_smoothing', 0.05)
    cw         = getattr(model, 'class_weights', None)
    num_classes = model.class_num if hasattr(model, 'class_num') else 86

    # Adaptive mixup
    if   epoch < 20:  mixup_alpha, do_mixup = 0.0, False
    elif epoch < 50:  mixup_alpha, do_mixup = 0.1, use_mixup
    elif epoch < 100: mixup_alpha, do_mixup = 0.2, use_mixup
    else:             mixup_alpha, do_mixup = 0.3, use_mixup

    for start in range(0, n, batch_size):
        batch_idx = idx[start: start + batch_size]
        left  = torch.from_numpy(train_x_left[batch_idx]).long().to(device)
        right = torch.from_numpy(train_x_right[batch_idx]).long().to(device)

        if label_type == 'multi_class':
            y = train_y[batch_idx].squeeze(-1).to(device)
        else:
            y = train_y[batch_idx].to(device)

        optimizer.zero_grad()

        # ---- HierDDI forward ----
        if _is_hier(model):
            out = model(left, right, labels=y,
                        return_aux_loss=True,
                        use_mixup=do_mixup,
                        mixup_alpha=mixup_alpha)

            if len(out) == 5:
                logits, super_logits, dis_loss, ncm_loss, mixed_labels = out
            else:
                logits, super_logits, dis_loss, ncm_loss = out
                mixed_labels = None

            if label_type == 'multi_class':
                if mixed_labels is not None:
                    task_loss = soft_cross_entropy(logits, mixed_labels, cw)
                    # Super-class loss: lấy argmax super từ routing
                    routing = F.softmax(model.hier_proto.leaf_to_super, dim=-1)  # (C, S)
                    mixed_super = mixed_labels @ routing                          # (B, S)
                    super_loss = soft_cross_entropy(super_logits, mixed_super)
                else:
                    smooth_y = label_smooth(y, num_classes, smoothing, device)
                    task_loss = soft_cross_entropy(logits, smooth_y, cw)
                    # Super-class soft target từ leaf assignment
                    routing = F.softmax(model.hier_proto.leaf_to_super, dim=-1)
                    super_target = smooth_y @ routing
                    super_loss = soft_cross_entropy(super_logits, super_target)

                loss = (task_loss
                        + model.lambda_super * super_loss
                        + model.lambda_dis   * dis_loss
                        + model.lambda_ncm   * ncm_loss)
            else:
                loss = loss_func(logits, y)

        # ---- Fallback cho model khác ----
        else:
            logits = model(left, right)
            loss = loss_func(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / max(1, n // batch_size)
    print(f'[Fold {fold}][Epoch {epoch}] train loss: {avg_loss:.4f}')


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@torch.no_grad()
def test(
    model, loss_func,
    test_x_left, test_x_right, test_y,
    fold, epoch, batch_size, label_type, device,
):
    model.eval()
    n = test_x_left.shape[0]
    all_logits, all_labels = [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        left  = torch.from_numpy(test_x_left[start:end]).long().to(device)
        right = torch.from_numpy(test_x_right[start:end]).long().to(device)

        if label_type == 'multi_class':
            y = test_y[start:end].squeeze(-1).to(device)
        else:
            y = test_y[start:end].to(device)

        # Eval: chỉ lấy leaf logits
        if _is_hier(model):
            logits = model(left, right)
            if isinstance(logits, tuple):
                logits = logits[0]
        else:
            logits = model(left, right)

        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if label_type == 'multi_class':
        preds     = all_logits.argmax(-1).numpy()
        labels_np = all_labels.squeeze(-1).numpy()
        acc       = accuracy_score(labels_np, preds)
        f1        = f1_score(labels_np, preds, average='macro', zero_division=0)
        precision = precision_score(labels_np, preds, average='macro', zero_division=0)
        recall    = recall_score(labels_np, preds, average='macro', zero_division=0)
        kappa     = cohen_kappa_score(labels_np, preds)
        score = [acc, f1, precision, recall, kappa]
        print(f'[Fold {fold}][Epoch {epoch}] acc:{acc:.3f} f1:{f1:.3f} '
              f'precision:{precision:.3f} recall:{recall:.3f} kappa:{kappa:.3f}')

    elif label_type == 'binary_class':
        probs     = torch.sigmoid(all_logits).squeeze(-1).numpy()
        preds     = (probs > 0.5).astype(int)
        labels_np = all_labels.squeeze(-1).numpy().astype(int)
        acc       = accuracy_score(labels_np, preds)
        f1        = f1_score(labels_np, preds, average='macro', zero_division=0)
        precision = precision_score(labels_np, preds, average='macro', zero_division=0)
        recall    = recall_score(labels_np, preds, average='macro', zero_division=0)
        try:    auc = roc_auc_score(labels_np, probs)
        except: auc = 0.0
        score = [acc, f1, precision, recall, auc]
        print(f'[Fold {fold}][Epoch {epoch}] acc:{acc:.3f} f1:{f1:.3f} '
              f'precision:{precision:.3f} recall:{recall:.3f} auc:{auc:.3f}')

    else:
        probs     = torch.sigmoid(all_logits).numpy()
        preds     = (probs > 0.5).astype(int)
        labels_np = all_labels.numpy().astype(int)
        acc       = accuracy_score(labels_np, preds)
        f1        = f1_score(labels_np, preds, average='macro', zero_division=0)
        precision = precision_score(labels_np, preds, average='macro', zero_division=0)
        recall    = recall_score(labels_np, preds, average='macro', zero_division=0)
        score = [acc, f1, precision, recall]
        print(f'[Fold {fold}][Epoch {epoch}] acc:{acc:.3f} f1:{f1:.3f} '
              f'precision:{precision:.3f} recall:{recall:.3f}')

    return score