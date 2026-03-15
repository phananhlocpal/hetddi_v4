"""
Training loop cho AdvancedHetDDI.
Thêm contrastive_weight vào tổng loss để align mol ↔ KG embedding.

Cách dùng:
    model = AdvancedHetDDI(...)
    trainer = AdvancedTrainer(model, optimizer, class_num, device, contrastive_weight=0.1)

    for epoch in range(num_epochs):
        loss = trainer.train_epoch(dataloader)
        acc, f1 = trainer.eval_epoch(val_dataloader)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import numpy as np


class AdvancedTrainer:
    def __init__(
        self,
        model,
        optimizer,
        class_num: int,
        device,
        contrastive_weight: float = 0.1,   # λ cho contrastive loss
        task: str = 'multiclass',           # 'multiclass' | 'multilabel'
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.contrastive_weight = contrastive_weight
        self.task = task
        self.class_num = class_num

        if task == 'multiclass':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    # ---------------------------------------------------------------
    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            left, right, labels = batch
            left = left.to(self.device)
            right = right.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # forward — yêu cầu contrastive loss
            logits, contrastive_loss = self.model(
                left, right,
                labels=labels,
                return_contrastive=True,
            )

            # task loss
            if self.task == 'multiclass':
                task_loss = self.criterion(logits, labels)
            else:
                task_loss = self.criterion(logits, labels.float())

            # tổng loss
            loss = task_loss + self.contrastive_weight * contrastive_loss

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    # ---------------------------------------------------------------
    @torch.no_grad()
    def eval_epoch(self, dataloader) -> tuple[float, float]:
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in dataloader:
            left, right, labels = batch
            left = left.to(self.device)
            right = right.to(self.device)

            logits = self.model(left, right)  # không cần contrastive khi eval

            if self.task == 'multiclass':
                preds = logits.argmax(dim=-1).cpu().numpy()
                labs = labels.numpy()
            else:
                preds = (logits.sigmoid() > 0.5).cpu().numpy()
                labs = labels.numpy()

            all_preds.append(preds)
            all_labels.append(labs)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        if self.task == 'multiclass':
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        else:
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return acc, f1


# ---------------------------------------------------------------
# Ví dụ sử dụng với 2 thuốc mới hoàn toàn (S2 scenario)
# ---------------------------------------------------------------
def eval_new_drug_pair(model, new_smiles_a: str, new_smiles_b: str, device):
    """
    Predict interaction giữa 2 thuốc chưa từng xuất hiện trong training.

    Workflow:
    1. Thêm 2 SMILES mới vào model.smiles (extend graph)
    2. Đánh dấu là unseen → model dùng mol path thay KG
    3. Inference

    Lưu ý: trong thực tế nên rebuild mol graph hoặc dùng
    inductive GNN không cần rebuild (e.g. MPNN với shared weights).
    """
    model.eval()

    # Index của 2 thuốc mới là drug_num và drug_num+1
    idx_a = model.drug_num      # placeholder — cần extend smiles list
    idx_b = model.drug_num + 1

    # Đánh dấu unseen để trigger mol fallback
    model.mark_unseen_drugs([idx_a, idx_b])

    with torch.no_grad():
        left = torch.tensor([idx_a], device=device)
        right = torch.tensor([idx_b], device=device)
        logits = model(left, right)
        probs = logits.softmax(dim=-1)

    return probs