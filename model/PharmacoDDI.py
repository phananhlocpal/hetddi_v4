"""
PharmaEnhancedHetDDI

Chiến lược: KHÔNG thay thế KG — thêm pharmacophore features như auxiliary signal.

Architecture:
  KG embedding (strong, transductive)        ──┐
  Mol GNN embedding (existing)               ──┤ concat → drug_emb (600-dim, như cũ)
                                               │
  Pharmacophore global features (19-dim MLP) ──┘ projected → 300-dim, fused via gate

Tổng drug_emb vẫn là 600-dim (kg=300, mol=300), nhưng mol_emb được augmented
bởi pharmacophore features trước khi fuse với KG.

Tại sao cách này tốt hơn mol-only:
1. Giữ KG prior — thông tin về protein targets, pathways không mất
2. Pharmacophore global features (MW, logP, TPSA, CYP proxies) bổ sung
   thông tin ADME mà mol GNN học được chậm
3. Overhead nhỏ: chỉ thêm 1 MLP 19→300 + 1 gate
4. Không thay đổi interface — drop-in replacement cho AdvancedHetDDI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.hgnn import HGNN
from model.mol import Mol
from model.AdvanceHetDDI import (
    CoAttention, PrototypeMemory, InfoNCELoss
)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Precompute 19-dim global pharmacophore features per drug
# (same as PharmacoDDI but standalone, no DGL graph needed)
# ---------------------------------------------------------------------------
def _global_pharma_features(smiles_list: list) -> torch.Tensor:
    """
    Returns (N, 19) tensor of global molecular descriptors.
    Falls back to zeros if RDKit unavailable or SMILES invalid.
    """
    if not HAS_RDKIT:
        return torch.zeros(len(smiles_list), 19)

    feats = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                feats.append(torch.zeros(19))
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                try:
                    mol.UpdatePropertyCache(strict=False)
                except Exception:
                    pass

            mw     = Descriptors.MolWt(mol) / 500.0
            logp   = (Crippen.MolLogP(mol) + 5) / 15.0
            tpsa   = rdMolDescriptors.CalcTPSA(mol) / 200.0
            hbd    = rdMolDescriptors.CalcNumHBD(mol) / 10.0
            hba    = rdMolDescriptors.CalcNumHBA(mol) / 15.0
            rot    = rdMolDescriptors.CalcNumRotatableBonds(mol) / 20.0
            rings  = rdMolDescriptors.CalcNumRings(mol) / 8.0
            arom   = rdMolDescriptors.CalcNumAromaticRings(mol) / 6.0
            fcsp3  = rdMolDescriptors.CalcFractionCSP3(mol)
            heavy  = mol.GetNumHeavyAtoms() / 60.0
            stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol) / 5.0
            hetero = rdMolDescriptors.CalcNumHeteroatoms(mol) / 20.0
            lip    = float(mw*500<=500 and logp*15-5<=5 and hbd*10<=5 and hba*15<=10)
            veber  = float(rot*20<=10 and tpsa*200<=140)
            bertz  = min(Descriptors.BertzCT(mol)/2000.0, 1.0)
            chi0   = min(Descriptors.Chi0(mol)/30.0, 1.0)
            smi_c  = Chem.MolToSmiles(mol).lower()
            cyp1a2 = float(any(a.GetAtomicNum()==7 and a.GetIsAromatic() for a in mol.GetAtoms()))
            cyp3a4 = float('c1cnc' in smi_c or 'c1ccnc' in smi_c)
            cyp2c9 = float('c1ccsc1' in smi_c)

            f = [mw, logp, tpsa, hbd, hba, rot, rings, arom, fcsp3,
                 heavy, stereo, hetero, lip, veber, bertz, chi0,
                 cyp1a2, cyp3a4, cyp2c9]
            feats.append(torch.tensor(f, dtype=torch.float32))
        except Exception:
            feats.append(torch.zeros(19))

    out = torch.stack(feats)   # (N, 19)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# PharmaEnhancedHetDDI
# ---------------------------------------------------------------------------
class PharmaEnhancedHetDDI(nn.Module):
    """
    AdvancedHetDDI + pharmacophore global feature augmentation.

    Thay đổi so với AdvancedHetDDI:
    - Precompute 19-dim global pharma descriptors cho mọi drug
    - Thêm pharma_enc: MLP(19 → 300) để project sang mol space
    - Thêm pharma_gate: học xem nên blend bao nhiêu pharma vào mol_emb
    - Mọi thứ khác giữ nguyên → kappa/acc dự kiến tốt hơn 2-3%
    """

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

        # ---- Pharmacophore global feature augmentation ----
        # Precompute once, store as buffer
        print("Computing pharmacophore global features...")
        pharma_feats = _global_pharma_features(smiles)   # (N, 19)
        self.register_buffer('_pharma_feats', pharma_feats)

        # Project 19-dim → mol_size (300)
        mol_dim = self.mol_size if mode in ('only_mol', 'concat') else self.kg_size
        self.pharma_enc = nn.Sequential(
            nn.Linear(19, mol_dim // 2),
            nn.LayerNorm(mol_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mol_dim // 2, mol_dim),
            nn.LayerNorm(mol_dim),
            nn.GELU(),
        )

        # Learned gate: how much pharma features to blend into mol_emb
        self.pharma_gate = nn.Sequential(
            nn.Linear(mol_dim * 2, mol_dim),
            nn.Sigmoid()
        )
        self._mol_dim = mol_dim

        # ---- Single drug dim ----
        if mode == 'only_kg':
            single_dim = self.kg_size
        elif mode == 'only_mol':
            single_dim = self.mol_size
        else:
            single_dim = self.kg_size + self.mol_size

        # ---- Co-attention ----
        self.co_attn = CoAttention(single_dim, num_heads=num_attn_heads, dropout=dropout)

        # ---- Pair dim ----
        pair_dim = single_dim * 2

        # ---- Prototype memory ----
        self.proto_mem = PrototypeMemory(pair_dim, class_num)

        # ---- Decoder with residual ----
        decoder_in = pair_dim * 2
        hidden = decoder_in // 2
        self.dec_proj = nn.Linear(decoder_in, hidden)
        self.dec_layers = nn.ModuleList([
            self._make_dec_block(hidden, dropout) for _ in range(3)
        ])
        self.dec_head = nn.Linear(hidden, class_num, bias=False)

        # ---- Losses ----
        self.contrastive_weight = 0.05
        self.label_smoothing = 0.05
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
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)
        )

    # ------------------------------------------------------------------
    def _cache_embeddings(self):
        if self.mode in ('only_kg', 'concat'):
            self._kg_emb = self.kg_fc(self.kg()[:self.drug_num])
        if self.mode in ('only_mol', 'concat'):
            mol_raw = self.mol_fc(self.mol())          # (N, mol_size)
            # Augment mol embedding with pharmacophore features
            pharma_emb = self.pharma_enc(self._pharma_feats)   # (N, mol_dim)
            mol_raw = torch.nan_to_num(mol_raw, nan=0.0, posinf=1e4, neginf=-1e4)
            pharma_emb = torch.nan_to_num(pharma_emb, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = self.pharma_gate(
                torch.cat([mol_raw, pharma_emb], dim=-1)
            )                                                   # (N, mol_dim)
            self._mol_emb = torch.nan_to_num(
                mol_raw + gate * pharma_emb, nan=0.0, posinf=1e4, neginf=-1e4
            )                                                    # gated addition

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
        return_contrastive: bool = False,
        use_mixup: bool = False,
    ):
        self._cache_embeddings()

        left_emb  = self._get_drug_emb(left)
        right_emb = self._get_drug_emb(right)

        # Co-attention
        left_emb, right_emb = self.co_attn(left_emb, right_emb)

        # Pair embedding
        pair_emb = torch.cat([left_emb, right_emb], dim=-1)

        # Prototype EMA update
        if self.training and labels is not None:
            label_idx = labels.argmax(-1) if labels.dim() > 1 else labels
            self.proto_mem.update_prototype(pair_emb.detach(), label_idx)

        # Mixup
        mixed_labels = None
        if self.training and use_mixup and labels is not None:
            lam = torch.distributions.Beta(0.2, 0.2).sample().to(pair_emb.device)
            idx = torch.randperm(pair_emb.size(0), device=pair_emb.device)
            pair_emb = lam * pair_emb + (1-lam) * pair_emb[idx]
            labels_oh = F.one_hot(labels, self.class_num).float() if labels.dim()==1 else labels.float()
            mixed_labels = lam * labels_oh + (1-lam) * labels_oh[idx]

        # Prototype feature
        proto_feat = self.proto_mem(pair_emb)

        # Decode
        logits = self._decode(torch.cat([pair_emb, proto_feat], dim=-1))

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
                kg_z  = self._kg_emb[known_idx]
                contrastive_loss = self.contrastive_loss_fn(mol_z, kg_z)

        if mixed_labels is not None:
            return logits, contrastive_loss, mixed_labels
        return logits, contrastive_loss

    # ------------------------------------------------------------------
    def mark_unseen_drugs(self, unseen_indices: list):
        self.kg_known_mask[unseen_indices] = False