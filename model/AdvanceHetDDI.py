"""
AdvancedHetDDI — Drug-Drug Interaction Prediction
====================================================
Improvements over HetDDI:
  1. Graph Transformer  : Transformer-style attention over molecular graphs
  2. KG Transformer     : Relational attention over the knowledge graph
  3. Contrastive Pre-training : InfoNCE loss on augmented molecular views
  4. Cross-modal Attention Fusion : Attend SMILES ↔ KG embeddings per drug
  5. Bilinear Interaction Predictor : F_A^T W F_B instead of plain MLP concat

Pipeline (per forward pass):
  SMILES  ──► Graph Transformer  ──► S_A / S_B        (molecular embedding)
  KG      ──► KG Transformer     ──► G_A / G_B        (biological embedding)
  S_A+G_A ──► Cross Attention    ──► F_A              (fused embedding)
  S_B+G_B ──► Cross Attention    ──► F_B
  F_A, F_B ──► Bilinear Head     ──► logits           (interaction score)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATv2Conv
from dgl import DGLGraph
from typing import Optional


# ─────────────────────────────────────────────
# 1. Graph Transformer  (molecular graph)
# ─────────────────────────────────────────────

class GraphTransformerLayer(nn.Module):
    """
    Single Graph Transformer layer.
    Uses multi-head attention between atom nodes (GATv2)
    followed by a Feed-Forward Network + LayerNorm.
    """

    def __init__(self, hidden: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden % num_heads == 0
        self.attn = GATv2Conv(
            in_feats=hidden,
            out_feats=hidden // num_heads,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            activation=None,
            share_weights=False,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.drop  = nn.Dropout(dropout)

    def forward(self, g: DGLGraph, h: torch.Tensor) -> torch.Tensor:
        # Multi-head attention  [N, heads, head_dim] → [N, hidden]
        h_attn = self.attn(g, h).flatten(1)
        h = self.norm1(h + self.drop(h_attn))
        h = self.norm2(h + self.drop(self.ffn(h)))
        return h


class MolGraphTransformer(nn.Module):
    """
    Stacks multiple GraphTransformerLayers then mean-pools atoms
    to produce a single drug embedding S ∈ R^hidden.
    """

    def __init__(
        self,
        atom_in_feats: int,
        hidden: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(atom_in_feats, hidden)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(hidden, num_heads, dropout) for _ in range(num_layers)]
        )
        self.pool_norm = nn.LayerNorm(hidden)

    def forward(self, g: DGLGraph, atom_feats: torch.Tensor) -> torch.Tensor:
        """
        g          : batched DGLGraph of molecular graphs
        atom_feats : [total_atoms, atom_in_feats]
        returns    : [num_graphs, hidden]
        """
        h = self.input_proj(atom_feats)
        for layer in self.layers:
            h = layer(g, h)
        # Mean pooling over atoms per molecule
        g.ndata['h'] = h
        emb = dgl.mean_nodes(g, 'h')
        return self.pool_norm(emb)


# ─────────────────────────────────────────────
# 2. KG Transformer  (relational knowledge graph)
# ─────────────────────────────────────────────

class RelationalGraphTransformerLayer(nn.Module):
    """
    Relational Graph Transformer layer.
    Projects each relation type into a separate key/value transform
    so that 'inhibits', 'metabolized_by', 'treats', etc. are handled
    with relation-specific weights.
    """

    def __init__(
        self,
        hidden: int,
        num_relations: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = hidden // num_heads
        self.hidden     = hidden

        self.q_proj = nn.Linear(hidden, hidden)
        # Per-relation key and value projections
        self.k_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_relations)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_relations)])
        self.out_proj = nn.Linear(hidden, hidden)

        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.drop  = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        node_feats: torch.Tensor,        # [N, hidden]
        edge_index: torch.Tensor,        # [2, E]  (src, dst)
        edge_type:  torch.Tensor,        # [E]     relation id
    ) -> torch.Tensor:
        N, H = node_feats.shape
        n_heads, head_dim = self.num_heads, self.head_dim

        Q = self.q_proj(node_feats).view(N, n_heads, head_dim)

        # Aggregate neighbour messages per node
        agg = torch.zeros_like(node_feats)          # [N, H]
        cnt = torch.zeros(N, 1, device=node_feats.device)

        for rel_id, (k_lin, v_lin) in enumerate(zip(self.k_proj, self.v_proj)):
            mask = edge_type == rel_id
            if mask.sum() == 0:
                continue
            src = edge_index[0][mask]
            dst = edge_index[1][mask]

            K_r = k_lin(node_feats[src]).view(-1, n_heads, head_dim)  # [E_r, heads, d]
            V_r = v_lin(node_feats[src]).view(-1, n_heads, head_dim)

            # Attention score  [E_r, heads]
            attn = (Q[dst] * K_r).sum(-1) * self.scale
            attn = F.softmax(attn, dim=0)                             # softmax over edges per dst

            # Weighted value  [E_r, hidden]
            msg = (attn.unsqueeze(-1) * V_r).reshape(-1, H)

            agg.index_add_(0, dst, msg)
            cnt.index_add_(0, dst, torch.ones(mask.sum(), 1, device=cnt.device))

        cnt = cnt.clamp(min=1)
        agg = self.out_proj(agg / cnt)

        node_feats = self.norm1(node_feats + self.drop(agg))
        node_feats = self.norm2(node_feats + self.drop(self.ffn(node_feats)))
        return node_feats


class KGTransformer(nn.Module):
    """
    Encodes the entire knowledge graph; returns embeddings for all nodes.
    Drug node embeddings are extracted by index after encoding.
    """

    def __init__(
        self,
        node_in_feats: int,
        hidden: int,
        num_relations: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_in_feats, hidden)
        self.layers = nn.ModuleList([
            RelationalGraphTransformerLayer(hidden, num_relations, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        node_feats: torch.Tensor,   # [N_kg, node_in_feats]
        edge_index: torch.Tensor,   # [2, E]
        edge_type:  torch.Tensor,   # [E]
    ) -> torch.Tensor:
        h = self.input_proj(node_feats)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
        return self.norm(h)                 # [N_kg, hidden]


# ─────────────────────────────────────────────
# 3. Contrastive Pre-training (InfoNCE)
# ─────────────────────────────────────────────

class MolecularAugmenter:
    """
    Two stochastic views of a molecular graph:
      - view1: randomly drop drop_rate fraction of edges
      - view2: independently drop drop_rate fraction of edges
    """

    def __init__(self, drop_rate: float = 0.10):
        self.drop_rate = drop_rate

    def __call__(self, g: DGLGraph):
        """Returns two augmented DGLGraph copies."""
        return self._drop_edges(g), self._drop_edges(g)

    def _drop_edges(self, g: DGLGraph) -> DGLGraph:
        num_edges = g.num_edges()
        mask = torch.rand(num_edges, device=g.device) > self.drop_rate
        eids  = torch.where(mask)[0]
        return dgl.edge_subgraph(g, eids, relabel_nodes=False, store_ids=False)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE / NT-Xent loss.
    z1, z2 : [B, D]  — two views of the same molecule
    Diagonal entries are positive pairs; off-diagonal are negatives.
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = torch.mm(z1, z2.T) / temperature          # [B, B]
    labels = torch.arange(B, device=z1.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss


# ─────────────────────────────────────────────
# 4. Cross-modal Attention Fusion
# ─────────────────────────────────────────────

class CrossModalAttentionFusion(nn.Module):
    """
    Fuses molecular embedding S and KG embedding G for one drug.

    Mechanism:
        Query  = S   (structural view)
        Key    = G   (biological view)
        Value  = G

    Output F = concat( S , Attention(S→G) ) projected to hidden.
    """

    def __init__(self, hidden: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, S: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        S : [B, hidden]  molecular embedding
        G : [B, hidden]  KG embedding
        returns F : [B, hidden]
        """
        # Unsqueeze seq dim  → [B, 1, hidden]
        S_seq = S.unsqueeze(1)
        G_seq = G.unsqueeze(1)

        attended, _ = self.cross_attn(query=S_seq, key=G_seq, value=G_seq)
        attended = attended.squeeze(1)                        # [B, hidden]

        # Residual + concat fusion
        fused = self.proj(torch.cat([S, attended], dim=-1))   # [B, hidden]
        return self.norm(fused + S)                           # residual from S


# ─────────────────────────────────────────────
# 5. Bilinear Interaction Predictor
# ─────────────────────────────────────────────

class BilinearInteractionHead(nn.Module):
    """
    score = F_A^T W F_B
    where W ∈ R^{hidden × hidden × class_num} is a rank-decomposed bilinear tensor.

    For multi-class DDI we use a separate W_k per class (low-rank factorisation).
    """

    def __init__(self, hidden: int, class_num: int, rank: int = 32, dropout: float = 0.1):
        super().__init__()
        # Low-rank factorisation: W_k ≈ U_k V_k^T
        self.U = nn.Parameter(torch.empty(class_num, hidden, rank))
        self.V = nn.Parameter(torch.empty(class_num, hidden, rank))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        self.bias = nn.Parameter(torch.zeros(class_num))
        self.drop  = nn.Dropout(dropout)

    def forward(self, F_A: torch.Tensor, F_B: torch.Tensor) -> torch.Tensor:
        """
        F_A, F_B : [B, hidden]
        returns   : [B, class_num]
        """
        # Project each drug into rank space per class
        # U : [C, H, R]  F_A : [B, H]
        # a_k = F_A @ U_k  → [B, R]
        a = torch.einsum('bh, chr -> bcr', self.drop(F_A), self.U)   # [B, C, R]
        b = torch.einsum('bh, chr -> bcr', self.drop(F_B), self.V)   # [B, C, R]

        # score_k = sum_r a_k * b_k
        scores = (a * b).sum(-1) + self.bias                          # [B, C]
        return scores


# ─────────────────────────────────────────────
# 6. AdvancedHetDDI  (main model)
# ─────────────────────────────────────────────

class AdvancedHetDDI(nn.Module):
    """
    AdvancedHetDDI
    ==============
    Args
    ----
    mol_graphs     : pre-built DGL batched molecular graph with atom features
                     stored in g.ndata['atom_feat']  [total_atoms, atom_in_feats]
    kg_node_feats  : Tensor [N_kg, node_in_feats]  — initial KG node features
    kg_edge_index  : Tensor [2, E]                 — KG edges (src, dst)
    kg_edge_type   : Tensor [E]                    — relation id per edge
    drug_node_ids  : LongTensor [num_drugs]        — which KG node indices are drugs
    atom_in_feats  : int
    node_in_feats  : int
    hidden         : int
    num_mol_layers : int
    num_kg_layers  : int
    num_relations  : int
    class_num      : int   (1 for binary, >1 for typed DDI)
    num_heads      : int
    bilinear_rank  : int   (rank for low-rank bilinear decomposition)
    dropout        : float
    """

    def __init__(
        self,
        mol_graphs:      DGLGraph,
        kg_node_feats:   torch.Tensor,
        kg_edge_index:   torch.Tensor,
        kg_edge_type:    torch.Tensor,
        drug_node_ids:   torch.Tensor,
        atom_in_feats:   int,
        node_in_feats:   int,
        hidden:          int   = 256,
        num_mol_layers:  int   = 3,
        num_kg_layers:   int   = 2,
        num_relations:   int   = 10,
        class_num:       int   = 1,
        num_heads:       int   = 4,
        bilinear_rank:   int   = 32,
        dropout:         float = 0.1,
    ):
        super().__init__()

        # ── stored tensors ──────────────────────────────────────────────
        self.register_buffer('kg_node_feats', kg_node_feats)
        self.register_buffer('kg_edge_index', kg_edge_index)
        self.register_buffer('kg_edge_type',  kg_edge_type)
        self.register_buffer('drug_node_ids', drug_node_ids)
        self.mol_graphs = mol_graphs          # DGLGraph (batched molecules)
        self.class_num  = class_num

        # ── 1. Molecular Graph Transformer ──────────────────────────────
        self.mol_encoder = MolGraphTransformer(
            atom_in_feats=atom_in_feats,
            hidden=hidden,
            num_layers=num_mol_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── 2. KG Transformer ───────────────────────────────────────────
        self.kg_encoder = KGTransformer(
            node_in_feats=node_in_feats,
            hidden=hidden,
            num_relations=num_relations,
            num_layers=num_kg_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── 3. Contrastive pre-training components ───────────────────────
        self.augmenter = MolecularAugmenter(drop_rate=0.10)
        self.proj_head = nn.Sequential(           # projection for InfoNCE
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
        )

        # ── 4. Cross-modal Attention Fusion ─────────────────────────────
        self.fusion = CrossModalAttentionFusion(
            hidden=hidden,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── 5. Bilinear Interaction Predictor ────────────────────────────
        self.predictor = BilinearInteractionHead(
            hidden=hidden,
            class_num=max(class_num, 1),
            rank=bilinear_rank,
            dropout=dropout,
        )

    # ── helpers ─────────────────────────────────────────────────────────

    def _encode_mol(self, g: Optional[DGLGraph] = None) -> torch.Tensor:
        """Returns [num_drugs, hidden] molecular embeddings."""
        graph = g if g is not None else self.mol_graphs
        atom_feats = graph.ndata['atom_feat']
        return self.mol_encoder(graph, atom_feats)       # [num_drugs, hidden]

    def _encode_kg(self) -> torch.Tensor:
        """Returns [num_drugs, hidden] KG embeddings for drug nodes."""
        all_emb = self.kg_encoder(
            self.kg_node_feats, self.kg_edge_index, self.kg_edge_type
        )                                                # [N_kg, hidden]
        return all_emb[self.drug_node_ids]               # [num_drugs, hidden]

    # ── contrastive pre-training step ───────────────────────────────────

    def contrastive_step(self) -> torch.Tensor:
        """
        Call this during pre-training.
        Returns InfoNCE loss over all drugs in mol_graphs.
        """
        g1, g2 = self.augmenter(self.mol_graphs)
        z1 = self.proj_head(self._encode_mol(g1))
        z2 = self.proj_head(self._encode_mol(g2))
        return contrastive_loss(z1, z2)

    # ── main forward ────────────────────────────────────────────────────

    def forward(
        self,
        left:  torch.LongTensor,   # [B]  drug indices for drug A
        right: torch.LongTensor,   # [B]  drug indices for drug B
    ) -> torch.Tensor:
        """
        Returns logits [B] (binary) or [B, class_num] (typed DDI).

        Pipeline:
          SMILES → mol_encoder  → S             [num_drugs, H]
          KG     → kg_encoder   → G             [num_drugs, H]
          S+G    → fusion       → F             [num_drugs, H]
          F_A, F_B → predictor  → logits        [B (, C)]
        """
        # ── encode ──────────────────────────────────────────────────────
        S = self._encode_mol()           # [num_drugs, hidden]
        G = self._encode_kg()            # [num_drugs, hidden]

        # ── fuse per drug ────────────────────────────────────────────────
        # fusion expects batch dimension → process all drugs at once
        F_all = self.fusion(S, G)        # [num_drugs, hidden]

        # ── select drug pairs ────────────────────────────────────────────
        F_A = F_all[left]               # [B, hidden]
        F_B = F_all[right]              # [B, hidden]

        # ── predict interaction ──────────────────────────────────────────
        logits = self.predictor(F_A, F_B)   # [B, C]

        if self.class_num == 1:
            return logits.squeeze(-1)        # [B]
        return logits                        # [B, C]

