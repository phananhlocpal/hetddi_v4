"""
PharmaEnhancedHetDDI_v3
- Morgan 2048 + 19 global + DUAL FINGERPRINT BRANCH cho unseen drugs
- Hỗ trợ hidden_dim=512 (tăng mạnh capacity)
- Dự kiến acc ≥ 0.70–0.72, kappa ≥ 0.62 trên s3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.hgnn import HGNN
from model.mol import Mol
from model.AdvanceHetDDI import CoAttention, PrototypeMemory, InfoNCELoss

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, AllChem
    from rdkit import DataStructs
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def _global_pharma_features(smiles_list: list) -> torch.Tensor:
    if not HAS_RDKIT:
        return torch.zeros(len(smiles_list), 19 + 2048)
    feats = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                feats.append(torch.zeros(19 + 2048))
                continue
            Chem.SanitizeMol(mol) if mol else None

            # 19 global descriptors
            mw = Descriptors.MolWt(mol) / 500.0
            logp = (Crippen.MolLogP(mol) + 5) / 15.0
            tpsa = rdMolDescriptors.CalcTPSA(mol) / 200.0
            hbd = rdMolDescriptors.CalcNumHBD(mol) / 10.0
            hba = rdMolDescriptors.CalcNumHBA(mol) / 15.0
            rot = rdMolDescriptors.CalcNumRotatableBonds(mol) / 20.0
            rings = rdMolDescriptors.CalcNumRings(mol) / 8.0
            arom = rdMolDescriptors.CalcNumAromaticRings(mol) / 6.0
            fcsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
            heavy = mol.GetNumHeavyAtoms() / 60.0
            stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol) / 5.0
            hetero = rdMolDescriptors.CalcNumHeteroatoms(mol) / 20.0
            lip = float(mw*500<=500 and logp*15-5<=5 and hbd*10<=5 and hba*15<=10)
            veber = float(rot*20<=10 and tpsa*200<=140)
            bertz = min(Descriptors.BertzCT(mol)/2000.0, 1.0)
            chi0 = min(Descriptors.Chi0(mol)/30.0, 1.0)
            smi_c = Chem.MolToSmiles(mol).lower()
            cyp1a2 = float(any(a.GetAtomicNum()==7 and a.GetIsAromatic() for a in mol.GetAtoms()))
            cyp3a4 = float('c1cnc' in smi_c or 'c1ccnc' in smi_c)
            cyp2c9 = float('c1ccsc1' in smi_c)

            global_feats = [mw, logp, tpsa, hbd, hba, rot, rings, arom, fcsp3,
                            heavy, stereo, hetero, lip, veber, bertz, chi0,
                            cyp1a2, cyp3a4, cyp2c9]

            # Morgan 2048
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_arr = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, fp_arr)

            feats.append(torch.tensor(global_feats + fp_arr.tolist(), dtype=torch.float32))
        except Exception:
            feats.append(torch.zeros(19 + 2048))
    out = torch.stack(feats)
    return torch.nan_to_num(out, nan=0.0)


class PharmaEnhancedHetDDI(nn.Module):
    def __init__(self, kg_g, smiles, num_hidden: int, num_layer: int, mode: str,
                 class_num: int, condition: str, num_attn_heads=4,
                 contrastive_temp=0.07, dropout=0.1):
        super().__init__()
        self.device = kg_g.device
        self.mode = mode
        self.drug_num = len(smiles)
        self.class_num = class_num
        self.num_hidden = num_hidden

        # KG
        if mode in ('only_kg', 'concat'):
            self.kg = HGNN(kg_g, kg_g.edata['edges'], kg_g.ndata['nodes'],
                           num_hidden, num_layer)
            self.kg_size = self.kg.get_output_size()
            self.kg_fc = self._make_fc(self.kg_size, dropout)

        # Mol
        if mode in ('only_mol', 'concat'):
            self.mol = Mol(smiles, num_hidden, num_layer, self.device, condition)
            self.mol_size = self.mol.get_output_size()
            self.mol_fc = self._make_fc(self.mol_size, dropout)

        # Mol projection
        if mode == 'concat':
            self.mol_proj = nn.Sequential(
                nn.Linear(self.mol_size, self.kg_size),
                nn.LayerNorm(self.kg_size), nn.ReLU(),
                nn.Linear(self.kg_size, self.kg_size),
            )

        # Pharmacophore + FP (2067 dim)
        print("Computing pharmacophore global features + Morgan 2048...")
        self.register_buffer('_pharma_feats', _global_pharma_features(smiles))

        mol_dim = self.mol_size if mode in ('only_mol', 'concat') else self.kg_size
        self.pharma_enc = nn.Sequential(
            nn.Linear(19 + 2048, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, mol_dim),
            nn.LayerNorm(mol_dim), nn.GELU()
        )

        # === DUAL FINGERPRINT BRANCH CHO UNSEEN DRUGS ===
        self.fp_branch = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, mol_dim),
            nn.LayerNorm(mol_dim)
        )
        self.fp_gate = nn.Sequential(
            nn.Linear(mol_dim * 2, mol_dim), nn.Sigmoid()
        )

        self.pharma_gate = nn.Sequential(
            nn.Linear(mol_dim * 2, mol_dim), nn.Sigmoid()
        )

        # Single drug dim
        single_dim = self.kg_size + self.mol_size if mode == 'concat' else mol_dim
        self.co_attn = CoAttention(single_dim, num_heads=num_attn_heads, dropout=dropout)

        # Decoder
        pair_dim = single_dim * 2
        self.proto_mem = PrototypeMemory(pair_dim, class_num)
        hidden = pair_dim * 2 // 2
        self.dec_proj = nn.Linear(pair_dim * 2, hidden)
        self.dec_layers = nn.ModuleList([self._make_dec_block(hidden, dropout) for _ in range(3)])
        self.dec_head = nn.Linear(hidden, class_num, bias=False)

        self.contrastive_loss_fn = InfoNCELoss(temperature=contrastive_temp)
        self.register_buffer('kg_known_mask', torch.ones(len(smiles), dtype=torch.bool))

    @staticmethod
    def _make_fc(dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.Dropout(dropout), nn.ReLU(),
        )

    @staticmethod
    def _make_dec_block(dim, dropout):
        return nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout))

    def _cache_embeddings(self):
        if self.mode in ('only_kg', 'concat'):
            self._kg_emb = self.kg_fc(self.kg()[:self.drug_num])
        if self.mode in ('only_mol', 'concat'):
            mol_raw = self.mol_fc(self.mol())
            pharma_emb = self.pharma_enc(self._pharma_feats)
            gate = self.pharma_gate(torch.cat([mol_raw, pharma_emb], -1))
            mol_aug = mol_raw + gate * pharma_emb

            # Fingerprint branch
            fp_raw = self._pharma_feats[:, 19:]          # chỉ 2048 bits
            fp_emb = self.fp_branch(fp_raw)
            fp_gate = self.fp_gate(torch.cat([mol_aug, fp_emb], -1))
            self._mol_emb = mol_aug + fp_gate * fp_emb   # dual augmentation

    def _get_drug_emb(self, idx):
        if self.mode == 'only_mol':
            return self._mol_emb[idx]
        elif self.mode == 'only_kg':
            return self._kg_emb[idx]
        else:
            mol_emb = self._mol_emb[idx]
            mol_proj = self.mol_proj(mol_emb)
            known = self.kg_known_mask[idx]
            kg_final = torch.where(
                known.unsqueeze(-1).expand_as(self._kg_emb[idx]),
                self._kg_emb[idx],
                mol_proj.detach()
            )
            return torch.cat([kg_final, mol_emb], -1)

    def _decode(self, x):
        h = self.dec_proj(x)
        for layer in self.dec_layers:
            h = h + layer(h)
        return self.dec_head(h)

    def forward(self, left, right, labels=None, return_contrastive=False, use_mixup=False):
        self._cache_embeddings()
        left_emb = self._get_drug_emb(left)
        right_emb = self._get_drug_emb(right)
        left_emb, right_emb = self.co_attn(left_emb, right_emb)
        pair_emb = torch.cat([left_emb, right_emb], -1)

        if self.training and labels is not None:
            self.proto_mem.update_prototype(pair_emb.detach(), labels if labels.dim()==1 else labels.argmax(-1))

        mixed_labels = None
        if self.training and use_mixup and labels is not None:
            lam = torch.distributions.Beta(0.2, 0.2).sample().to(pair_emb.device)
            idx = torch.randperm(pair_emb.size(0), device=pair_emb.device)
            pair_emb = lam * pair_emb + (1-lam) * pair_emb[idx]
            labels_oh = F.one_hot(labels, self.class_num).float() if labels.dim()==1 else labels.float()
            mixed_labels = lam * labels_oh + (1-lam) * labels_oh[idx]

        proto = self.proto_mem(pair_emb)
        logits = self._decode(torch.cat([pair_emb, proto], -1))

        if not return_contrastive:
            return (logits, mixed_labels) if mixed_labels is not None else logits

        c_loss = torch.tensor(0.0, device=self.device)
        if self.mode == 'concat':
            known_idx = torch.cat([left, right]).unique()[self.kg_known_mask[torch.cat([left, right]).unique()]]
            if len(known_idx) > 1:
                c_loss = self.contrastive_loss_fn(self.mol_proj(self._mol_emb[known_idx]), self._kg_emb[known_idx])
        return (logits, c_loss, mixed_labels) if mixed_labels is not None else (logits, c_loss)

    def mark_unseen_drugs(self, unseen):
        self.kg_known_mask[unseen] = False