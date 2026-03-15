import torch
import torch.nn as nn
import torch.nn.functional as F
from model.hgnn import HGNN
from model.mol import Mol


# ---------------------------------------------------------------------------
# Co-attention đơn giản hơn — không dùng gate phức tạp
# Giữ lại residual chuẩn, thêm FFN sau attention (như Transformer block)
# ---------------------------------------------------------------------------
class CoAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_a2b = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_a1 = nn.LayerNorm(dim)
        self.norm_b1 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.norm_b2 = nn.LayerNorm(dim)
        # FFN sau attention — giúp transform features sau khi đã cross-attend
        self.ffn_a = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout)
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout)
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        a_seq = a.unsqueeze(1)
        b_seq = b.unsqueeze(1)
        # Cross-attention + residual + layernorm (Pre-LN style)
        a_ctx, _ = self.attn_a2b(self.norm_a1(a_seq), self.norm_b1(b_seq), self.norm_b1(b_seq))
        b_ctx, _ = self.attn_b2a(self.norm_b1(b_seq), self.norm_a1(a_seq), self.norm_a1(a_seq))
        a = a + a_ctx.squeeze(1)
        b = b + b_ctx.squeeze(1)
        # FFN + residual
        a = a + self.ffn_a(self.norm_a2(a))
        b = b + self.ffn_b(self.norm_b2(b))
        return a, b


# ---------------------------------------------------------------------------
# Prototype memory — giữ nguyên từ v1, fix init
# ---------------------------------------------------------------------------
class PrototypeMemory(nn.Module):
    def __init__(self, dim: int, class_num: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(class_num, dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.dim = dim
        self.class_num = class_num

    def forward(self, pair_emb: torch.Tensor) -> torch.Tensor:
        pair_norm = F.normalize(pair_emb, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = pair_norm @ proto_norm.t()
        weights = F.softmax(sim / 0.1, dim=-1)
        return weights @ self.prototypes

    @torch.no_grad()
    def update_prototype(self, pair_emb: torch.Tensor, labels: torch.Tensor, momentum: float = 0.99):
        for c in range(self.class_num):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            mean_emb = pair_emb[mask].mean(0)
            self.prototypes.data[c] = (
                momentum * self.prototypes.data[c] + (1 - momentum) * mean_emb
            )


# ---------------------------------------------------------------------------
# InfoNCE
# ---------------------------------------------------------------------------
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, mol_emb: torch.Tensor, kg_emb: torch.Tensor) -> torch.Tensor:
        B = mol_emb.size(0)
        z1 = F.normalize(mol_emb, dim=-1)
        z2 = F.normalize(kg_emb, dim=-1)
        logits = z1 @ z2.t() / self.tau
        labels = torch.arange(B, device=z1.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


# ---------------------------------------------------------------------------
# AdvancedHetDDI v3
#
# Thay đổi so với v2:
# 1. Bỏ Focal Loss — dùng lại CrossEntropy + label_smoothing nhẹ (0.05)
#    Focal Loss làm train loss giảm quá nhanh → overfit
# 2. Co-attention dùng Pre-LN + FFN thay vì gate phức tạp
#    Pre-LN ổn định hơn trong training, FFN giúp transform sau cross-attend
# 3. Bỏ symmetric pair (a+b, a*b) — quay về concat(a,b)
#    a*b tạo gradient instability với embedding lớn
# 4. Thêm Mixup augmentation trong pair embedding space
#    Giúp model generalize tốt hơn với unseen drug pairs (s2/s3)
# 5. Decoder dùng 4 layer thay vì 3, với residual connection
#    Tăng capacity mà không tăng width → ít overfit hơn
# ---------------------------------------------------------------------------
class AdvancedHetDDI(nn.Module):
    def __init__(
        self,
        kg_g,
        smiles,
        num_hidden: int,
        num_layer: int,
        mode: str,
        class_num: int,
        condition: str,
        num_attn_heads: int = 4,
        contrastive_temp: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.smiles = smiles
        self.device = kg_g.device
        self.mode = mode
        self.drug_num = len(smiles)
        self.class_num = class_num
        self.num_hidden = num_hidden

        # ---- KG encoder ----
        if mode in ('only_kg', 'concat'):
            self.kg = HGNN(
                kg_g, kg_g.edata['edges'], kg_g.ndata['nodes'],
                num_hidden, num_layer=num_layer
            )
            self.kg_size = self.kg.get_output_size()
            self.kg_fc = self._make_fc(self.kg_size, dropout)

        # ---- Mol encoder ----
        if mode in ('only_mol', 'concat'):
            self.mol = Mol(smiles, num_hidden, num_layer, self.device, condition)
            self.mol_size = self.mol.gnn.get_output_size()
            self.mol_fc = self._make_fc(self.mol_size, dropout)

        # ---- Mol projection (contrastive + KG fallback) ----
        if mode == 'concat':
            self.mol_proj = nn.Sequential(
                nn.Linear(self.mol_size, self.kg_size),
                nn.LayerNorm(self.kg_size),
                nn.ReLU(),
                nn.Linear(self.kg_size, self.kg_size),
            )

        # ---- Single drug dim ----
        if mode == 'only_kg':
            single_dim = self.kg_size
        elif mode == 'only_mol':
            single_dim = self.mol_size
        else:
            single_dim = self.kg_size + self.mol_size

        # ---- Co-attention ----
        self.co_attn = CoAttention(single_dim, num_heads=num_attn_heads, dropout=dropout)

        # ---- Pair dim: concat(a, b) ----
        pair_dim = single_dim * 2

        # ---- Prototype memory ----
        self.proto_mem = PrototypeMemory(pair_dim, class_num)

        # ---- Decoder với residual connections ----
        # Input: pair_emb (pair_dim) + proto_feat (pair_dim) = pair_dim*2
        decoder_in = pair_dim * 2
        hidden = decoder_in // 2

        self.dec_proj = nn.Linear(decoder_in, hidden)  # project xuống trước
        self.dec_layers = nn.ModuleList([
            self._make_dec_block(hidden, dropout),
            self._make_dec_block(hidden, dropout),
            self._make_dec_block(hidden, dropout),
        ])
        self.dec_head = nn.Linear(hidden, class_num, bias=False)

        # ---- Contrastive ----
        self.contrastive_weight = 0.05
        self.contrastive_loss_fn = InfoNCELoss(temperature=contrastive_temp)

        # ---- KG known mask ----
        self.register_buffer(
            'kg_known_mask',
            torch.ones(len(smiles), dtype=torch.bool)
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _make_fc(dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.Dropout(dropout), nn.ReLU(),
        )

    @staticmethod
    def _make_dec_block(dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    # ------------------------------------------------------------------
    def _get_drug_emb(self, drug_indices: torch.Tensor) -> torch.Tensor:
        if self.mode == 'only_mol':
            return self._mol_emb[drug_indices]
        elif self.mode == 'only_kg':
            return self._kg_emb[drug_indices]
        else:
            mol_emb = self._mol_emb[drug_indices]
            mol_projected = self.mol_proj(mol_emb)
            known = self.kg_known_mask[drug_indices]
            kg_selected = self._kg_emb[drug_indices]
            kg_final = torch.where(
                known.unsqueeze(-1).expand_as(kg_selected),
                kg_selected,
                mol_projected.detach(),
            )
            return torch.cat([kg_final, mol_emb], dim=-1)

    # ------------------------------------------------------------------
    def _cache_embeddings(self):
        if self.mode in ('only_kg', 'concat'):
            self._kg_emb = self.kg_fc(self.kg()[:self.drug_num])
        if self.mode in ('only_mol', 'concat'):
            self._mol_emb = self.mol_fc(self.mol())

    # ------------------------------------------------------------------
    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decoder với residual connections."""
        h = self.dec_proj(x)
        for layer in self.dec_layers:
            h = h + layer(h)   # residual
        return self.dec_head(h)

    # ------------------------------------------------------------------
    def _mixup_pair(
        self,
        pair_emb: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 0.2,
    ):
        """
        Mixup trong pair embedding space.
        Trộn 2 sample ngẫu nhiên với lam ~ Beta(alpha, alpha).
        Chỉ dùng khi training.
        Trả về (mixed_pair_emb, mixed_labels_soft).
        """
        B = pair_emb.size(0)
        lam = torch.distributions.Beta(alpha, alpha).sample().to(pair_emb.device)
        idx = torch.randperm(B, device=pair_emb.device)

        mixed_emb = lam * pair_emb + (1 - lam) * pair_emb[idx]

        # Soft labels: one-hot trộn
        if labels.dim() == 1:
            labels_oh = F.one_hot(labels, self.class_num).float()
        else:
            labels_oh = labels.float()
        mixed_labels = lam * labels_oh + (1 - lam) * labels_oh[idx]

        return mixed_emb, mixed_labels

    # ------------------------------------------------------------------
    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        labels: torch.Tensor = None,
        return_contrastive: bool = False,
        use_mixup: bool = False,
    ):
        self._cache_embeddings()

        left_emb = self._get_drug_emb(left)
        right_emb = self._get_drug_emb(right)

        # Co-attention
        left_emb, right_emb = self.co_attn(left_emb, right_emb)

        # Pair embedding: simple concat
        pair_emb = torch.cat([left_emb, right_emb], dim=-1)

        # Prototype EMA update
        if self.training and labels is not None:
            label_idx = labels.argmax(dim=-1) if labels.dim() > 1 else labels
            self.proto_mem.update_prototype(pair_emb.detach(), label_idx)

        # Mixup augmentation (optional, chỉ training)
        mixed_labels = None
        if self.training and use_mixup and labels is not None:
            pair_emb, mixed_labels = self._mixup_pair(pair_emb, labels)

        # Prototype feature
        proto_feat = self.proto_mem(pair_emb)

        # Decode
        decoder_input = torch.cat([pair_emb, proto_feat], dim=-1)
        logits = self._decode(decoder_input)

        if not return_contrastive:
            if mixed_labels is not None:
                return logits, mixed_labels
            return logits

        # Contrastive loss
        contrastive_loss = torch.tensor(0.0, device=self.device)
        if self.mode == 'concat':
            all_indices = torch.cat([left, right]).unique()
            known = self.kg_known_mask[all_indices]
            if known.sum() > 1:
                known_idx = all_indices[known]
                mol_z = self.mol_proj(self._mol_emb[known_idx])
                kg_z = self._kg_emb[known_idx]
                contrastive_loss = self.contrastive_loss_fn(mol_z, kg_z)

        if mixed_labels is not None:
            return logits, contrastive_loss, mixed_labels
        return logits, contrastive_loss

    # ------------------------------------------------------------------
    def mark_unseen_drugs(self, unseen_indices: list):
        self.kg_known_mask[unseen_indices] = False