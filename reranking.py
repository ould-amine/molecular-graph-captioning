import math
import pickle
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from retrieval_answer import *
from train import *
from data_utils import *

INPUT_DIR = "/kaggle/input/data-kaggle"
OUTPUT_DIR = "/kaggle/working"

TRAIN_GRAPHS = os.path.join(INPUT_DIR, "train_graphs.pkl")
VAL_GRAPHS   = os.path.join(INPUT_DIR, "validation_graphs.pkl")
TEST_GRAPHS  = os.path.join(INPUT_DIR, "test_graphs.pkl")
TRAIN_EMB_CSV = os.path.join(OUTPUT_DIR, "train_embeddings.csv")
VAL_EMB_CSV   = os.path.join(OUTPUT_DIR, "validation_embeddings.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def graph_signature(g):
    """
    Signature structurelle enrichie pour mieux coller aux descriptions SciBERT.
    """
    n_nodes = int(g.num_nodes)
    e_dir = int(g.edge_index.size(1))
    n_edges = max(1, e_dir // 2)
    atomic_nums = g.x[:, 0]
    n_oxygen = torch.sum(atomic_nums == 8).item()
    n_nitrogen = torch.sum(atomic_nums == 7).item()
    n_aromatic = torch.sum(g.x[:, 7] == 1).item() 
    
    density = n_edges / max(1, n_nodes)
    
    return {
        'size': (n_nodes, n_edges),
        'chem': (n_oxygen, n_nitrogen, n_aromatic),
        'density': density
    }

def structure_similarity(sig1, sig2):
    """
    Compare deux signatures pro.
    """
    dist_size = abs(math.log((sig1['size'][0] + 1) / (sig2['size'][0] + 1))) + \
                0.5 * abs(math.log((sig1['size'][1] + 1) / (sig2['size'][1] + 1)))
    
    dist_chem = abs(sig1['chem'][0] - sig2['chem'][0]) + \
                abs(sig1['chem'][1] - sig2['chem'][1]) + \
                0.5 * abs(sig1['chem'][2] - sig2['chem'][2])
    
    dist_dens = abs(sig1['density'] - sig2['density'])
    
    total_dist = dist_size + 0.8 * dist_chem + 0.4 * dist_dens
    return 1.0 / (1.0 + total_dist)

def plot_validation_heatmap(val_mol_embs, val_text_embs, n_samples=10):
    sim_matrix = torch.matmul(val_mol_embs[:n_samples], val_text_embs[:n_samples].t())
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix.cpu().numpy(), annot=True, cmap='YlGnBu')
    plt.title(f'Validation Similarity Heatmap (Ground Truth on Diagonal)')
    plt.xlabel('Ground Truth Descriptions (Indices)')
    plt.ylabel('Validation Molecules (Indices)')
    # Sauvegarde dans le répertoire de travail Kaggle
    plt.savefig(os.path.join(OUTPUT_DIR, 'validation_heatmap.png'))
    plt.show()



@torch.no_grad()
def run_validation_diagnostic(model, val_data_path, val_emb_dict, device, n_samples=10):
    model.eval()
    # Chargement des données de validation
    val_ds = PreprocessedGraphDataset(val_data_path, val_emb_dict)
    val_dl = DataLoader(val_ds, batch_size=n_samples, shuffle=False, collate_fn=collate_fn)
    
    # Récupération d'un batch pour la heatmap
    graphs, text_embs = next(iter(val_dl))
    graphs = graphs.to(device)
    text_embs = text_embs.to(device)
    
    # Encodage
    mol_embs = model(graphs) 
    text_embs = F.normalize(text_embs, dim=-1)
    
    print("Generating validation heatmap...")
    plot_validation_heatmap(mol_embs, text_embs, n_samples=n_samples)
    
@torch.no_grad()
def retrieve_ensemble_soft_ranking(models_list, train_data, test_data, train_emb_dict, device, output_csv,
                                  top_k=20, alpha=0.2):
    """
    Ensemble Retrieval via Soft Ranking
    """
    # 1. Préparer les textes (communs à tous les modèles)
    train_id2desc = load_descriptions_from_graphs(train_data)
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    # 2. Charger les graphes de test
    test_ds = PreprocessedGraphDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 3. Calculer les matrices de similarité pour CHAQUE modèle
    # similarities_total sera la somme des scores (Soft Ranking)
    similarities_total = None
    
    for model_info in models_list:
        name = model_info['name']
        model = model_info['model']
        weight = model_info.get('weight', 1.0)
        model.eval()
        
        print(f"En cours d'encodage avec le modèle : {name}...")
        current_mol_embs = []
        for graphs in test_dl:
            graphs = graphs.to(device)
            # Certains modèles (Graphormer) ont besoin de calculs spécifiques
            # mais ici on suppose qu'ils gèrent leur forward en interne
            mol_emb = model(graphs) 
            current_mol_embs.append(mol_emb)
        
        current_mol_embs = torch.cat(current_mol_embs, dim=0)
        
        # Similarité cosinus pour ce modèle
        current_sims = current_mol_embs @ train_embs.t()
        
        if similarities_total is None:
            similarities_total = current_sims * weight
        else:
            similarities_total += current_sims * weight

    # 4. Reranking structurel sur la similarité fusionnée
    print("Calcul des signatures structurelles...")
    test_sigs = [graph_signature(g) for g in test_ds.graphs]
    with open(train_data, "rb") as f:
        train_graphs = pickle.load(f)
    train_id2sig = {g.id: graph_signature(g) for g in train_graphs}
    train_sigs = [train_id2sig[tid] for tid in train_ids]

    results = []
    test_ids_ordered = test_ds.ids
    
    for i, test_id in enumerate(test_ids_ordered):
        sims = similarities_total[i]
        topk = sims.topk(top_k) # On peut augmenter top_k pour l'ensemble
        topk_idx = topk.indices.tolist()

        best_score = -1e18
        best_train_id = None

        for idx in topk_idx:
            soft_sim = sims[idx].item()
            struct_sim = structure_similarity(test_sigs[i], train_sigs[idx])
            
            # Score final mixant l'Ensemble Soft Ranking + Structure
            score = soft_sim + alpha * struct_sim

            if score > best_score:
                best_score = score
                best_train_id = train_ids[idx]

        results.append({"ID": test_id, "description": train_id2desc[best_train_id]})

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f" Ensemble Soft Ranking terminé. Sauvegardé sous {output_csv}")


