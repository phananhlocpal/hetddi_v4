"""
AdvancedHetDDI v2
=================
Improved architecture for Drug–Drug Interaction prediction.

Key differences vs previous version:

1. Atom-level embeddings kept as sequences
2. KG neighbor embeddings used instead of single vector
3. Real cross-attention between atoms and KG neighbors
4. Subgraph KG encoding
5. Neural Tensor Interaction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATv2Conv


# ============================================================
# 1. Atom Graph Transformer
# ============================================================

class AtomGraphEncoder(nn.Module):

    def __init__(self, atom_dim, hidden, layers=3, heads=4):
        super().__init__()

        self.input_proj = nn.Linear(atom_dim, hidden)

        self.layers = nn.ModuleList([
            GATv2Conv(
                hidden,
                hidden // heads,
                num_heads=heads
            )
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(hidden)

    def forward(self, g):

        h = g.ndata["atom_feat"]
        h = self.input_proj(h)

        for layer in self.layers:
            h = layer(g, h).flatten(1)

        g.ndata["h"] = h

        # keep atom sequence
        atom_seq = dgl.unbatch(g)

        seq_list = []
        for mol in atom_seq:
            seq_list.append(mol.ndata["h"])

        return seq_list
    
# ============================================================
# 2. KG Transformer
# ============================================================

class KGEncoder(nn.Module):

    def __init__(self, node_dim, hidden, relations):

        super().__init__()

        self.node_proj = nn.Linear(node_dim, hidden)

        self.rel_emb = nn.Embedding(relations, hidden)

        self.attn = nn.MultiheadAttention(
            hidden,
            num_heads=4,
            batch_first=True
        )

    def forward(self, node_feats, edge_index, edge_type, drug_nodes):

        node_feats = self.node_proj(node_feats)

        neighbor_embeddings = []

        for drug in drug_nodes:

            mask = edge_index[0] == drug
            neigh = edge_index[1][mask]

            if len(neigh) == 0:
                neighbor_embeddings.append(
                    node_feats[drug].unsqueeze(0)
                )
                continue

            neigh_feat = node_feats[neigh]

            neighbor_embeddings.append(neigh_feat)

        return neighbor_embeddings
    

# ============================================================
# 3. Cross Modal Fusion
# ============================================================

class CrossModalFusion(nn.Module):

    def __init__(self, hidden):

        super().__init__()

        self.cross = nn.MultiheadAttention(
            hidden,
            num_heads=4,
            batch_first=True
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.norm = nn.LayerNorm(hidden)

    def forward(self, atom_seq, kg_seq):

        fused_embeddings = []

        for atoms, kg in zip(atom_seq, kg_seq):

            atoms = atoms.unsqueeze(0)
            kg = kg.unsqueeze(0)

            attn, _ = self.cross(
                query=atoms,
                key=kg,
                value=kg
            )

            fused = atoms + attn

            pooled = fused.mean(dim=1)

            fused_embeddings.append(pooled.squeeze(0))

        return torch.stack(fused_embeddings)

# ============================================================
# 4. Neural Tensor Interaction
# ============================================================

class NeuralTensorInteraction(nn.Module):

    def __init__(self, hidden, k=16):

        super().__init__()

        self.W = nn.Parameter(
            torch.randn(k, hidden, hidden)
        )

        self.V = nn.Linear(hidden * 2, k)

        self.out = nn.Linear(k, 1)

    def forward(self, A, B):

        batch = A.size(0)

        tensor_scores = []

        for i in range(self.W.size(0)):
            score = torch.sum(
                (A @ self.W[i]) * B,
                dim=1
            )
            tensor_scores.append(score)

        tensor_scores = torch.stack(tensor_scores, dim=1)

        linear = self.V(torch.cat([A, B], dim=1))

        h = torch.tanh(tensor_scores + linear)

        return self.out(h).squeeze(-1)
    
# ============================================================
# 5. AdvancedHetDDI
# ============================================================

class AdvancedHetDDIv2(nn.Module):

    def __init__(
        self,
        atom_dim,
        kg_dim,
        hidden,
        relations
    ):

        super().__init__()

        self.atom_encoder = AtomGraphEncoder(
            atom_dim,
            hidden
        )

        self.kg_encoder = KGEncoder(
            kg_dim,
            hidden,
            relations
        )

        self.fusion = CrossModalFusion(hidden)

        self.predictor = NeuralTensorInteraction(hidden)

    def encode_drugs(
        self,
        mol_graph,
        kg_node_feats,
        edge_index,
        edge_type,
        drug_nodes
    ):

        atom_seq = self.atom_encoder(mol_graph)

        kg_seq = self.kg_encoder(
            kg_node_feats,
            edge_index,
            edge_type,
            drug_nodes
        )

        drug_embed = self.fusion(atom_seq, kg_seq)

        return drug_embed

    def forward(
        self,
        mol_graph,
        kg_node_feats,
        edge_index,
        edge_type,
        drug_nodes,
        left,
        right
    ):

        drug_embed = self.encode_drugs(
            mol_graph,
            kg_node_feats,
            edge_index,
            edge_type,
            drug_nodes
        )

        A = drug_embed[left]
        B = drug_embed[right]

        score = self.predictor(A, B)

        return score