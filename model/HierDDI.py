"""
HierDDI — Breakthrough Architecture for Drug-Drug Interaction Prediction
=========================================================================

Core insight cho s3 (cả 2 thuốc unseen):
  Model cần học RELATIONSHIP SPACE, không chỉ drug embeddings.

3 đột phá kiến trúc so với AdvancedHetDDI:

1. DUAL-STREAM DISENTANGLEMENT
   - Mỗi drug embedding được tách thành 2 phần: "structure code" (bất biến, từ mol)
     và "context code" (từ KG neighborhood)
   - Contrastive loss buộc structure code align với context code TRONG CÙNG drug
   - Khi test với unseen drug: structure code là anchor đáng tin cậy

2. INTERACTION GRAPH ATTENTION (IGA)
   - Thay vì chỉ cross-attend A↔B, xây dựng mini graph: [A, B, pair_node]
   - pair_node nhận message từ cả A và B qua GAT
   - pair_node output là interaction-aware representation
   - Khác biệt với CoAttention: pair_node học interaction pattern,
     không chỉ "A nhìn B" mà "A và B cùng tạo ra interaction node"

3. HIERARCHICAL PROTOTYPE TREE
   - Thay vì flat prototypes (86 class × dim), dùng 2-level tree:
     Level 1: ~12 "super-classes" (nhóm DDI theo cơ chế: pharmacokinetic, pharmacodynamic...)
     Level 2: 86 leaf classes
   - Prediction = routing qua tree: P(leaf) = P(super) × P(leaf|super)
   - Giúp model generalize: unseen drug pairs → ít nhất predict đúng super-class
   - Hierarchical contrastive: same super-class phải gần nhau trong embedding space

Author: AdvancedHetDDI v5 / HierDDI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.hgnn import HGNN
from model.mol import Mol


# ---------------------------------------------------------------------------
# 1. Disentangled Drug Encoder
# ---------------------------------------------------------------------------
class DisentangledEncoder(nn.Module):
    """
    Tách embedding thành structure_code và context_code.
    structure_code: invariant features từ molecular graph
    context_code:   relational features từ KG neighborhood

    Trong s3, khi thuốc unseen → chỉ có structure_code.
    Model học cách dùng structure_code như anchor.
    """
    def __init__(self, mol_dim: int, kg_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        # Project mol → structure code
        self.struct_proj = nn.Sequential(
            nn.Linear(mol_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        # Project KG → context code
        self.ctx_proj = nn.Sequential(
            nn.Linear(kg_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        # Bridge: khi KG available, fuse struct + context
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.hidden = hidden

    def forward(self, mol_emb: torch.Tensor, kg_emb: torch.Tensor = None,
                known_mask: torch.Tensor = None):
        """
        mol_emb:   (B, mol_dim)
        kg_emb:    (B, kg_dim) or None
        known_mask: (B,) bool — True nếu drug có trong KG

        Returns: (B, hidden) fused embedding
                 struct_code (B, hidden) — for contrastive loss
                 ctx_code    (B, hidden) — for contrastive loss
        """
        struct_code = self.struct_proj(mol_emb)  # (B, H)

        if kg_emb is not None and known_mask is not None:
            ctx_code = self.ctx_proj(kg_emb)     # (B, H)
            # Fuse nơi có KG, dùng struct_code nơi không có
            fused = self.fusion(torch.cat([struct_code, ctx_code], dim=-1))
            # Graceful: nơi unseen → dùng struct_code thay fused
            known = known_mask.unsqueeze(-1).float()
            out = known * fused + (1 - known) * struct_code
            return out, struct_code, ctx_code
        else:
            # Không có KG: chỉ dùng struct_code
            ctx_code = struct_code  # placeholder
            return struct_code, struct_code, ctx_code


# ---------------------------------------------------------------------------
# 2. Interaction Graph Attention (IGA)
# ---------------------------------------------------------------------------
class InteractionGraphAttention(nn.Module):
    """
    Mini 3-node graph: [drug_A, drug_B, interaction_node]
    interaction_node học representation của CẶP THUỐC,
    không chỉ là "A nhìn B" như trong CoAttention.

    Forward pass:
      Round 1: interaction_node ← attend(A, B)
      Round 2: A ← attend(A, interaction_node)
               B ← attend(B, interaction_node)
      Round 3: interaction_node ← attend(updated_A, updated_B)

    Output: updated_A, updated_B, interaction_node
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, num_rounds: int = 2):
        super().__init__()
        self.num_rounds = num_rounds

        # Interaction node init: learned from A+B
        self.init_interact = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )

        # Round attention modules
        # interact ← (A, B)
        self.attn_ab2i = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_rounds)
        ])
        # A ← (A, interact)
        self.attn_i2a = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_rounds)
        ])
        # B ← (B, interact)
        self.attn_i2b = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_rounds)
        ])

        # Layer norms
        self.norm_a  = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_rounds)])
        self.norm_b  = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_rounds)])
        self.norm_i  = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_rounds)])

        # FFN
        self.ffn_a = nn.ModuleList([self._make_ffn(dim, dropout) for _ in range(num_rounds)])
        self.ffn_b = nn.ModuleList([self._make_ffn(dim, dropout) for _ in range(num_rounds)])
        self.ffn_i = nn.ModuleList([self._make_ffn(dim, dropout) for _ in range(num_rounds)])

        self.norm_a2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_rounds)])
        self.norm_b2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_rounds)])
        self.norm_i2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_rounds)])

    @staticmethod
    def _make_ffn(dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        """a, b: (B, D)"""
        # Init interaction node từ A và B
        i = self.init_interact(torch.cat([a, b], dim=-1))  # (B, D)

        for r in range(self.num_rounds):
            # --- Update interaction node from A and B ---
            a_seq = a.unsqueeze(1)   # (B, 1, D)
            b_seq = b.unsqueeze(1)
            i_seq = i.unsqueeze(1)
            ab_seq = torch.cat([a_seq, b_seq], dim=1)  # (B, 2, D)  key/value

            i_new, _ = self.attn_ab2i[r](i_seq, ab_seq, ab_seq)
            i = self.norm_i[r](i + i_new.squeeze(1))
            i = self.norm_i2[r](i + self.ffn_i[r](i))

            # --- Update A from interaction node ---
            i_seq = i.unsqueeze(1)
            a_new, _ = self.attn_i2a[r](a_seq, i_seq, i_seq)
            a = self.norm_a[r](a + a_new.squeeze(1))
            a = self.norm_a2[r](a + self.ffn_a[r](a))
            a_seq = a.unsqueeze(1)

            # --- Update B from interaction node ---
            b_new, _ = self.attn_i2b[r](b_seq, i_seq, i_seq)
            b = self.norm_b[r](b + b_new.squeeze(1))
            b = self.norm_b2[r](b + self.ffn_b[r](b))
            b_seq = b.unsqueeze(1)

        return a, b, i


# ---------------------------------------------------------------------------
# 3. Hierarchical Prototype Tree
# ---------------------------------------------------------------------------
class HierarchicalPrototypeTree(nn.Module):
    """
    2-level prototype tree:
      Super-classes (level 1): num_super classes
      Leaf-classes  (level 2): class_num leaves

    Each leaf belongs to exactly one super-class.
    P(leaf | pair_emb) = P(super | pair_emb) × P(leaf | super, pair_emb)

    Benefits cho s3:
    - Unseen drug pairs → ít nhất predict đúng mechanism category
    - Hierarchical contrastive: drug pairs cùng super-class gần nhau
    - Smoother gradient: rare leaf classes học từ super-class signal
    """
    def __init__(self, dim: int, class_num: int, num_super: int = 12):
        super().__init__()
        self.class_num = class_num
        self.num_super = num_super

        # Super-class prototypes
        self.super_prototypes = nn.Parameter(torch.empty(num_super, dim))
        nn.init.xavier_uniform_(self.super_prototypes)

        # Leaf prototypes
        self.leaf_prototypes = nn.Parameter(torch.empty(class_num, dim))
        nn.init.xavier_uniform_(self.leaf_prototypes)

        # Learnable leaf→super assignment (soft, differentiable)
        # Mỗi leaf có distribution over super-classes
        self.leaf_to_super = nn.Parameter(torch.randn(class_num, num_super))

        # Query projection: pair_emb → query
        self.query_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.temperature_super = nn.Parameter(torch.tensor(0.1))
        self.temperature_leaf  = nn.Parameter(torch.tensor(0.1))

    def forward(self, pair_emb: torch.Tensor):
        """
        pair_emb: (B, D)
        Returns:
            logits: (B, class_num) — final prediction logits
            super_logits: (B, num_super) — for hierarchical loss
            proto_feat: (B, D) — enriched feature for decoder
        """
        q = self.query_proj(pair_emb)  # (B, D)

        # Level 1: Super-class similarity
        q_norm = F.normalize(q, dim=-1)
        s_norm = F.normalize(self.super_prototypes, dim=-1)
        super_sim = q_norm @ s_norm.t()                           # (B, num_super)
        tau_s = self.temperature_super.abs().clamp(min=0.01)
        super_logits = super_sim / tau_s
        super_weights = F.softmax(super_logits, dim=-1)           # (B, num_super)

        # Level 2: Leaf similarity
        l_norm = F.normalize(self.leaf_prototypes, dim=-1)
        leaf_sim = q_norm @ l_norm.t()                            # (B, class_num)
        tau_l = self.temperature_leaf.abs().clamp(min=0.01)
        leaf_weights = F.softmax(leaf_sim / tau_l, dim=-1)        # (B, class_num)

        # Leaf→super routing (differentiable)
        routing = F.softmax(self.leaf_to_super, dim=-1)           # (class_num, num_super)
        # Expected super-class for each leaf given query
        # leaf_score[b, c] = leaf_sim[b,c] + routing[c] · super_weights[b]
        super_boost = super_weights @ routing.t()                 # (B, class_num)
        leaf_score = leaf_sim + super_boost                       # (B, class_num)

        # Proto feature: weighted sum of super + leaf prototypes
        super_feat = super_weights @ self.super_prototypes        # (B, D)
        leaf_feat  = leaf_weights  @ self.leaf_prototypes         # (B, D)
        proto_feat = super_feat + leaf_feat                       # (B, D)

        return leaf_score, super_logits, proto_feat

    @torch.no_grad()
    def update_prototypes(self, pair_emb: torch.Tensor, labels: torch.Tensor,
                          super_labels: torch.Tensor = None, momentum: float = 0.99):
        for c in range(self.class_num):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            mean = pair_emb[mask].mean(0)
            self.leaf_prototypes.data[c] = momentum * self.leaf_prototypes.data[c] + (1 - momentum) * mean

        if super_labels is not None:
            for s in range(self.num_super):
                mask = (super_labels == s)
                if mask.sum() == 0:
                    continue
                mean = pair_emb[mask].mean(0)
                self.super_prototypes.data[s] = momentum * self.super_prototypes.data[s] + (1 - momentum) * mean


# ---------------------------------------------------------------------------
# Contrastive losses
# ---------------------------------------------------------------------------
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)
        z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / self.tau
        labels = torch.arange(B, device=z1.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


class DisentanglementLoss(nn.Module):
    """
    Align structure_code ↔ context_code của CÙNG drug (positive pair).
    Đẩy xa struct/ctx của các drugs khác nhau (negative pairs).
    → Buộc mol và KG encode cùng latent space.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.nce = InfoNCELoss(temperature)

    def forward(self, struct_codes: torch.Tensor, ctx_codes: torch.Tensor,
                known_mask: torch.Tensor) -> torch.Tensor:
        # Chỉ tính với known drugs (có cả mol và KG)
        if known_mask.sum() < 2:
            return torch.tensor(0.0, device=struct_codes.device)
        s = struct_codes[known_mask]
        c = ctx_codes[known_mask]
        return self.nce(s, c)


# ---------------------------------------------------------------------------
# HierDDI — Main Model
# ---------------------------------------------------------------------------
class HierDDI(nn.Module):
    """
    Breakthrough architecture cho s3 DDI prediction.

    Pipeline:
    1. DisentangledEncoder: mol_emb + kg_emb → struct_code + ctx_code + fused_emb
    2. InteractionGraphAttention: (drug_A, drug_B) → (A', B', interaction_node)
    3. HierarchicalPrototypeTree: pair → leaf_score + super_logits + proto_feat
    4. Decoder: concat(A', B', interaction_node, proto_feat) → class_num

    Loss = CE_leaf + λ_super * CE_super + λ_dis * DisentanglementLoss + λ_ncm * InfoNCE
    """

    NUM_SUPER = 12  # ~12 mechanism categories trong DrugBank

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
        iga_rounds: int = 2,
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

        # ---- Disentangled encoder ----
        if mode == 'concat':
            self.dis_enc = DisentangledEncoder(
                mol_dim=self.mol_size,
                kg_dim=self.kg_size,
                hidden=num_hidden,
                dropout=dropout,
            )
            drug_dim = num_hidden  # output dim của DisentangledEncoder
        elif mode == 'only_mol':
            # Không có KG → chỉ project mol
            self.dis_enc = nn.Sequential(
                nn.Linear(self.mol_size, num_hidden),
                nn.LayerNorm(num_hidden), nn.ReLU(),
            )
            drug_dim = num_hidden
        else:
            self.dis_enc = nn.Sequential(
                nn.Linear(self.kg_size, num_hidden),
                nn.LayerNorm(num_hidden), nn.ReLU(),
            )
            drug_dim = num_hidden

        # ---- Interaction Graph Attention ----
        self.iga = InteractionGraphAttention(
            dim=drug_dim,
            num_heads=num_attn_heads,
            dropout=dropout,
            num_rounds=iga_rounds,
        )

        # ---- Hierarchical Prototype Tree ----
        # Input: concat(A', B', interaction_node) = drug_dim * 3
        pair_dim = drug_dim * 3
        self.hier_proto = HierarchicalPrototypeTree(
            dim=pair_dim,
            class_num=class_num,
            num_super=self.NUM_SUPER,
        )

        # ---- Decoder ----
        # Input: pair_dim (A'+B'+interact) + proto_feat (pair_dim) = pair_dim * 2
        decoder_in = pair_dim * 2
        hidden_dec = decoder_in // 2
        self.dec_proj = nn.Linear(decoder_in, hidden_dec)
        self.dec_layers = nn.ModuleList([
            self._make_dec_block(hidden_dec, dropout) for _ in range(3)
        ])
        self.dec_head = nn.Linear(hidden_dec, class_num, bias=False)

        # ---- Loss functions ----
        self.dis_loss_fn     = DisentanglementLoss(temperature=0.07)
        self.ncm_loss_fn     = InfoNCELoss(temperature=0.07)

        # Loss weights — set bởi main.py
        self.lambda_super    = 0.3   # hierarchical super-class loss weight
        self.lambda_dis      = 0.1   # disentanglement loss weight
        self.lambda_ncm      = 0.05  # NCM contrastive weight
        self.label_smoothing = 0.05
        self.contrastive_weight = 0.05  # backward compat

        # Class weights (inverse freq)
        self.register_buffer('class_weights', torch.ones(class_num))
        self.register_buffer('kg_known_mask', torch.ones(len(smiles), dtype=torch.bool))

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
    def _cache_embeddings(self):
        if self.mode in ('only_kg', 'concat'):
            self._kg_emb = self.kg_fc(self.kg()[:self.drug_num])
        if self.mode in ('only_mol', 'concat'):
            self._mol_emb = self.mol_fc(self.mol())

    def _get_drug_emb(self, drug_indices: torch.Tensor):
        """
        Returns (fused_emb, struct_code, ctx_code) cho batch drug indices.
        """
        if self.mode == 'concat':
            mol = self._mol_emb[drug_indices]
            kg  = self._kg_emb[drug_indices]
            known = self.kg_known_mask[drug_indices]
            return self.dis_enc(mol, kg, known)
        elif self.mode == 'only_mol':
            mol = self._mol_emb[drug_indices]
            fused = self.dis_enc(mol)
            return fused, fused, fused
        else:
            kg = self._kg_emb[drug_indices]
            fused = self.dis_enc(kg)
            return fused, fused, fused

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dec_proj(x)
        for layer in self.dec_layers:
            h = h + layer(h)
        return self.dec_head(h)

    # ------------------------------------------------------------------
    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        labels: torch.Tensor = None,
        return_aux_loss: bool = False,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
    ):
        # 1. Cache
        self._cache_embeddings()

        # 2. Disentangled drug embeddings
        a_emb, a_struct, a_ctx = self._get_drug_emb(left)
        b_emb, b_struct, b_ctx = self._get_drug_emb(right)

        # 3. Interaction Graph Attention → A', B', interaction_node
        a_emb, b_emb, interact = self.iga(a_emb, b_emb)

        # 4. Pair representation = concat(A', B', interaction_node)
        pair_emb = torch.cat([a_emb, b_emb, interact], dim=-1)  # (B, 3D)

        # 5. Prototype EMA update
        if self.training and labels is not None:
            label_idx = labels.argmax(-1) if labels.dim() > 1 else labels
            self.hier_proto.update_prototypes(pair_emb.detach(), label_idx)

        # 6. Mixup (optional)
        mixed_labels = None
        if self.training and use_mixup and labels is not None:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().to(pair_emb.device)
            idx = torch.randperm(pair_emb.size(0), device=pair_emb.device)
            pair_emb = lam * pair_emb + (1 - lam) * pair_emb[idx]
            if labels.dim() == 1:
                labels_oh = F.one_hot(labels, self.class_num).float()
            else:
                labels_oh = labels.float()
            mixed_labels = lam * labels_oh + (1 - lam) * labels_oh[idx]

        # 7. Hierarchical prototype tree
        leaf_score, super_logits, proto_feat = self.hier_proto(pair_emb)

        # 8. Decode
        decoder_in = torch.cat([pair_emb, proto_feat], dim=-1)
        logits = self._decode(decoder_in)

        if not return_aux_loss:
            if mixed_labels is not None:
                return logits, mixed_labels
            return logits

        # 9. Auxiliary losses
        # Disentanglement: struct ↔ ctx alignment
        all_struct = torch.cat([a_struct, b_struct], dim=0)
        all_ctx    = torch.cat([a_ctx,    b_ctx],    dim=0)
        all_known  = torch.cat([
            self.kg_known_mask[left],
            self.kg_known_mask[right],
        ], dim=0)
        dis_loss = self.dis_loss_fn(all_struct, all_ctx, all_known)

        # NCM: mol → KG alignment (như trước)
        ncm_loss = torch.tensor(0.0, device=self.device)
        if self.mode == 'concat':
            all_idx = torch.cat([left, right]).unique()
            known = self.kg_known_mask[all_idx]
            if known.sum() > 1:
                kidx = all_idx[known]
                ncm_loss = self.ncm_loss_fn(
                    F.normalize(self._mol_emb[kidx], dim=-1),
                    F.normalize(self._kg_emb[kidx],  dim=-1),
                )

        if mixed_labels is not None:
            return logits, super_logits, dis_loss, ncm_loss, mixed_labels
        return logits, super_logits, dis_loss, ncm_loss

    def mark_unseen_drugs(self, unseen_indices: list):
        self.kg_known_mask[unseen_indices] = False