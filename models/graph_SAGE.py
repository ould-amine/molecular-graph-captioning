import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils import softmax
import matplotlib.pyplot as plt

class AtomEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        node_emb_dim = hidden // 5
        self.emb_atomic_num = nn.Embedding(119, node_emb_dim)
        self.emb_chirality = nn.Embedding(9, node_emb_dim)
        self.emb_degree = nn.Embedding(11, node_emb_dim)
        self.emb_formal_charge = nn.Embedding(12, node_emb_dim)
        self.emb_num_hs = nn.Embedding(9, node_emb_dim)
        self.emb_num_radical = nn.Embedding(5, node_emb_dim)
        self.emb_hybridization = nn.Embedding(8, node_emb_dim)
        self.emb_aromatic = nn.Embedding(2, node_emb_dim)
        self.emb_in_ring = nn.Embedding(2, node_emb_dim)
        self.node_proj = nn.Linear(9 * node_emb_dim, hidden)

    def forward(self, x):
        return self.node_proj(torch.cat([
            self.emb_atomic_num(x[:,0]), self.emb_chirality(x[:,1]), self.emb_degree(x[:,2]),
            self.emb_formal_charge(x[:,3]), self.emb_num_hs(x[:,4]), self.emb_num_radical(x[:,5]),
            self.emb_hybridization(x[:,6]), self.emb_aromatic(x[:,7]), self.emb_in_ring(x[:,8])
        ], dim=-1))


INPUT_DIR = "/kaggle/input/data-kaggle"
OUTPUT_DIR = "/kaggle/working"

BATCH_SIZE = 4
EPOCHS = 90 
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_GRAPHS = os.path.join(INPUT_DIR, "train_graphs.pkl")
VAL_GRAPHS   = os.path.join(INPUT_DIR, "validation_graphs.pkl")
TRAIN_EMB_CSV = os.path.join(OUTPUT_DIR, "train_embeddings.csv")
VAL_EMB_CSV   = os.path.join(OUTPUT_DIR, "validation_embeddings.csv")



class MolGNN(nn.Module):
    def __init__(self, hidden=300, out_dim=1536, layers=4): # out_dim mis à jour pour SciBERT+MPNet
        super().__init__()
        self.hidden = hidden

        node_emb_dim = hidden // 5
        self.emb_atomic_num = nn.Embedding(119, node_emb_dim)
        self.emb_chirality = nn.Embedding(9, node_emb_dim)
        self.emb_degree = nn.Embedding(11, node_emb_dim)
        self.emb_formal_charge = nn.Embedding(12, node_emb_dim)
        self.emb_num_hs = nn.Embedding(9, node_emb_dim)
        self.emb_num_radical = nn.Embedding(5, node_emb_dim)
        self.emb_hybridization = nn.Embedding(8, node_emb_dim)
        self.emb_aromatic = nn.Embedding(2, node_emb_dim)
        self.emb_in_ring = nn.Embedding(2, node_emb_dim)
        
        # Projection pour aligner les embeddings de nœuds
        self.node_proj = nn.Linear(9 * node_emb_dim, hidden)

        # Edge embeddings (caractéristiques des liaisons)
        edge_emb_dim = hidden // 5
        self.emb_bond_type = nn.Embedding(22, edge_emb_dim)
        self.emb_stereo = nn.Embedding(6, edge_emb_dim)
        self.emb_conjugated = nn.Embedding(2, edge_emb_dim)
        self.edge_proj = nn.Linear(3 * edge_emb_dim, hidden)

    
        self.v_node_emb = nn.Embedding(1, hidden)
        self.v_node_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, hidden)) 
            for _ in range(layers - 1)
        ])

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.batch_norms.append(nn.BatchNorm1d(hidden))

        self.att_mlp = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch


        h = self.node_proj(torch.cat([
            self.emb_atomic_num(x[:,0]), self.emb_chirality(x[:,1]), self.emb_degree(x[:,2]),
            self.emb_formal_charge(x[:,3]), self.emb_num_hs(x[:,4]), self.emb_num_radical(x[:,5]),
            self.emb_hybridization(x[:,6]), self.emb_aromatic(x[:,7]), self.emb_in_ring(x[:,8])
        ], dim=-1))

        e = self.edge_proj(torch.cat([
            self.emb_bond_type(edge_attr[:,0]), self.emb_stereo(edge_attr[:,1]), self.emb_conjugated(edge_attr[:,2])
        ], dim=-1))

        v_h = self.v_node_emb(torch.zeros(batch.num_graphs, dtype=torch.long, device=h.device))

        for i, conv in enumerate(self.convs):
            
            h = h + v_h[batch_idx]
            h_res = h
            h = conv(h, edge_index, e)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = h + h_res
            h = F.dropout(h, p=0.1, training=self.training)

            if i < len(self.v_node_mlp):
                v_h_temp = global_add_pool(h, batch_idx) + v_h
                v_h = F.dropout(self.v_node_mlp[i](v_h_temp), p=0.1, training=self.training)

        
        att_scores = self.att_mlp(h).squeeze(-1)
        att_weights = softmax(att_scores, batch_idx).unsqueeze(-1)
        
        g = torch.zeros((batch.num_graphs, h.size(-1)), device=h.device)
        g = g.index_add(0, batch_idx, h * att_weights)

        # Projection finale vers l'espace 1536 (SciBERT + MPNet)
        g = self.proj(g)
        return F.normalize(g, dim=-1)



from torch_geometric.nn import SAGEConv, global_mean_pool

class MolGraphSAGE(nn.Module):
    def __init__(self, hidden=300, out_dim=1536, layers=3):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(layers)])
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, batch):
        h = self.atom_encoder(batch.x)
        for conv in self.convs:
            h = F.relu(conv(h, batch.edge_index))
            h = F.dropout(h, p=0.1, training=self.training)
        
        g = global_mean_pool(h, batch.batch)
        return F.normalize(self.proj(g), dim=-1)


