import torch
import torch.nn as nn
import dgl
from rdkit import Chem
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.utils import (
    mol_to_bigraph,
    CanonicalAtomFeaturizer,   # stores under 'h' — 74 dims
    CanonicalBondFeaturizer,   # stores under 'e' — 12 dims
)
from dgl.nn.pytorch.glob import AvgPooling


class Mol(nn.Module):
    def __init__(self, smiles, num_hidden, num_layer, device='cuda:0', condition='s3'):
        super().__init__()
        self.device = device
        self.readout = AvgPooling()

        atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
        bond_featurizer = CanonicalBondFeaturizer(bond_data_field='e')

        graphs = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
            g = mol_to_bigraph(
                mol,
                add_self_loop=True,
                node_featurizer=atom_featurizer,
                edge_featurizer=bond_featurizer,
                canonical_atom_order=False,
            )
            graphs.append(g)

        self.mol_g = dgl.batch(graphs).to(self.device)

        # CanonicalAtomFeaturizer → 74 dims, CanonicalBondFeaturizer → 12 dims
        self.gnn = AttentiveFPGNN(
            node_feat_size=74,
            edge_feat_size=12,
            num_layers=num_layer,
            graph_feat_size=num_hidden,
            dropout=0.2,
        ).to(self.device)

        self.output_size = num_hidden

    def get_output_size(self):
        return self.output_size

    def forward(self):
        node_feats = self.gnn(
            self.mol_g,
            self.mol_g.ndata['h'],   # ✓ CanonicalAtomFeaturizer uses 'h'
            self.mol_g.edata['e'],   # ✓ CanonicalBondFeaturizer uses 'e'
        )
        return self.readout(self.mol_g, node_feats)