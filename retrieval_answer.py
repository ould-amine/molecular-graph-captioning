import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from models.encoder import GEncoder
from data_utils import PreprocessedGraphDataset, load_id2emb, load_descriptions_from_graphs
from torch_geometric.data import Batch

def collate_fn_test(batch):
    batch_graph = Batch.from_data_list(batch)
    return batch_graph

@torch.no_grad()
def retrieve_descriptions(model, train_data_pkl, test_data_pkl, train_emb_dict, device, output_csv):
    train_id2desc = load_descriptions_from_graphs(train_data_pkl)
    
    train_ids = [str(k) for k in train_emb_dict.keys()]
    train_embs = torch.stack([train_emb_dict[int(id_)] for id_ in train_ids]).to(device)
    train_embs = model.project_text(train_embs)
    train_embs = F.normalize(train_embs, dim=-1)

    test_ds = PreprocessedGraphDataset(test_data_pkl)
    test_dl = DataLoader(
        test_ds, 
        batch_size=64, 
        shuffle=False, 
        collate_fn=collate_fn_test
    )

    test_mol_embs = []
    test_ids_ordered = []
    
    for batch_graph in test_dl:
        batch_graph = batch_graph.to(device)
        z_graph, _, _ = model(batch_graph)
        test_mol_embs.append(z_graph)
        
        current_batch_size = batch_graph.num_graphs
        idx_start = len(test_ids_ordered)
        batch_ids = test_ds.ids[idx_start : idx_start + current_batch_size]
        test_ids_ordered.extend(batch_ids)

    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    test_mol_embs = F.normalize(test_mol_embs, dim=-1)

    similarities = test_mol_embs @ train_embs.t()
    most_similar_indices = similarities.argmax(dim=-1).cpu()

    results = []
    for i, test_id in enumerate(test_ids_ordered):
        train_idx = most_similar_indices[i].item()
        retrieved_train_id = train_ids[train_idx]
        retrieved_desc = train_id2desc[retrieved_train_id]

        results.append({
            'ID': test_id,
            'description': retrieved_desc
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    return results_df

if __name__ == "__main__":
    BASE_PATH = "."
    DATA_DIR = os.path.join(BASE_PATH, "data")
    
    train_pkl = os.path.join(DATA_DIR, "train_graphs.pkl")
    test_pkl  = os.path.join(DATA_DIR, "test_graphs.pkl")
    train_emb_csv = os.path.join(DATA_DIR, "train_embeddings.csv")
    model_path = os.path.join(BASE_PATH, "models/saved_model_gatv2_sbert.pth")
    output_csv = os.path.join(DATA_DIR, "test_retrieved_descriptions.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_emb = load_id2emb(train_emb_csv)

    model = GEncoder(text_emb_dim=1152).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    retrieve_descriptions(
        model=model,
        train_data_pkl=train_pkl,
        test_data_pkl=test_pkl,
        train_emb_dict=train_emb,
        device=device,
        output_csv=output_csv
    )