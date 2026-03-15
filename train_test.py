import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, cohen_kappa_score, roc_auc_score,
)
from model.AdvanceHetDDI import AdvancedHetDDI


def _is_advanced(model) -> bool:
    return isinstance(model, AdvancedHetDDI)


def _forward(model, left, right, labels=None, training=False, use_mixup=False):
    """
    Unified forward.
    Returns (logits, contrastive_loss, mixed_labels_or_None)
    """
    if _is_advanced(model) and training:
        out = model(left, right, labels=labels,
                    return_contrastive=True, use_mixup=use_mixup)
        if len(out) == 3:
            logits, c_loss, mixed_labels = out
        else:
            logits, c_loss = out
            mixed_labels = None
        return logits, c_loss, mixed_labels
    else:
        logits = model(left, right)
        return logits, torch.tensor(0.0, device=left.device), None


def soft_cross_entropy(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy với soft labels (sau mixup hoặc label smoothing)."""
    log_prob = F.log_softmax(logits, dim=-1)
    return -(soft_labels * log_prob).sum(dim=-1).mean()


def cross_entropy_with_smoothing(logits, labels, smoothing=0.05, num_classes=86):
    """CrossEntropy + label smoothing nhẹ."""
    labels_oh = F.one_hot(labels, num_classes).float()
    smooth_labels = labels_oh * (1 - smoothing) + smoothing / num_classes
    return soft_cross_entropy(logits, smooth_labels)


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

    c_weight = getattr(model, 'contrastive_weight', 0.0)
    # Mixup chỉ bật sau epoch 20 (model cần học basic patterns trước)
    do_mixup = use_mixup and (epoch >= 20)

    for start in range(0, n, batch_size):
        batch_idx = idx[start: start + batch_size]
        left  = torch.from_numpy(train_x_left[batch_idx]).long().to(device)
        right = torch.from_numpy(train_x_right[batch_idx]).long().to(device)

        if label_type == 'multi_class':
            y = train_y[batch_idx].squeeze(-1).to(device)
        else:
            y = train_y[batch_idx].to(device)

        optimizer.zero_grad()

        logits, c_loss, mixed_labels = _forward(
            model, left, right, labels=y, training=True, use_mixup=do_mixup
        )

        # Loss
        if label_type == 'multi_class':
            if mixed_labels is not None:
                # Mixup đã tạo soft labels
                task_loss = soft_cross_entropy(logits, mixed_labels)
            else:
                # Label smoothing nhẹ
                task_loss = cross_entropy_with_smoothing(
                    logits, y,
                    smoothing=getattr(model, 'label_smoothing', 0.05),
                    num_classes=model.class_num if hasattr(model, 'class_num') else 86
                )
        else:
            task_loss = loss_func(logits, y)

        loss = task_loss + c_weight * c_loss

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

        logits, _, _ = _forward(model, left, right, labels=None, training=False)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if label_type == 'multi_class':
        preds = all_logits.argmax(dim=-1).numpy()
        labels_np = all_labels.squeeze(-1).numpy()

        acc       = accuracy_score(labels_np, preds)
        f1        = f1_score(labels_np, preds, average='macro', zero_division=0)
        precision = precision_score(labels_np, preds, average='macro', zero_division=0)
        recall    = recall_score(labels_np, preds, average='macro', zero_division=0)
        kappa     = cohen_kappa_score(labels_np, preds)

        score = [acc, f1, precision, recall, kappa]
        print(
            f'[Fold {fold}][Epoch {epoch}] '
            f'acc:{acc:.3f} f1:{f1:.3f} precision:{precision:.3f} '
            f'recall:{recall:.3f} kappa:{kappa:.3f}'
        )

    elif label_type == 'binary_class':
        probs  = torch.sigmoid(all_logits).squeeze(-1).numpy()
        preds  = (probs > 0.5).astype(int)
        labels_np = all_labels.squeeze(-1).numpy().astype(int)

        acc       = accuracy_score(labels_np, preds)
        f1        = f1_score(labels_np, preds, average='macro', zero_division=0)
        precision = precision_score(labels_np, preds, average='macro', zero_division=0)
        recall    = recall_score(labels_np, preds, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(labels_np, probs)
        except ValueError:
            auc = 0.0

        score = [acc, f1, precision, recall, auc]
        print(
            f'[Fold {fold}][Epoch {epoch}] '
            f'acc:{acc:.3f} f1:{f1:.3f} precision:{precision:.3f} '
            f'recall:{recall:.3f} auc:{auc:.3f}'
        )

    else:
        probs  = torch.sigmoid(all_logits).numpy()
        preds  = (probs > 0.5).astype(int)
        labels_np = all_labels.numpy().astype(int)

        acc       = accuracy_score(labels_np, preds)
        f1        = f1_score(labels_np, preds, average='macro', zero_division=0)
        precision = precision_score(labels_np, preds, average='macro', zero_division=0)
        recall    = recall_score(labels_np, preds, average='macro', zero_division=0)

        score = [acc, f1, precision, recall]
        print(
            f'[Fold {fold}][Epoch {epoch}] '
            f'acc:{acc:.3f} f1:{f1:.3f} precision:{precision:.3f} recall:{recall:.3f}'
        )

    return score