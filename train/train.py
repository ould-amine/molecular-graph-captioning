import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from models.encoder import GEncoder
from utils.data_utils import PreprocessedGraphDataset, collate_fn, load_id2emb

# ======================================
# Configuration
# ======================================
class Config:
    epochs = 50
    batch_size = 64
    learning_rate = 5e-4
    weight_decay = 1e-5
    temperature = 0.07  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = "." 
    model_name = "saved_model_gatv2_sbert.pth"

def contrastive_loss_improved(z_graph, z_text, temperature=0.07):
    """
    Calcule la perte contrastive symétrique avec normalisation L2.
    """
    z_graph = F.normalize(z_graph, dim=-1)
    z_text = F.normalize(z_text, dim=-1)
    logits = torch.matmul(z_graph, z_text.t()) / temperature
    labels = torch.arange(z_graph.size(0), device=z_graph.device)
    
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.t(), labels)
    
    return (loss_g2t + loss_t2g) / 2

# ======================================
# Main Training Function
# ======================================
def train():
    cfg = Config()
    
   
    train_csv = os.path.join(cfg.base_path, "data", "train_embeddings.csv")
    val_csv   = os.path.join(cfg.base_path, "data", "validation_embeddings.csv")
    train_pkl = os.path.join(cfg.base_path, "data", "train_graphs.pkl")
    val_pkl   = os.path.join(cfg.base_path, "data", "validation_graphs.pkl")

    print("Loading precomputed embeddings...")
    train_emb = load_id2emb(train_csv)
    val_emb   = load_id2emb(val_csv)

    train_dataset = PreprocessedGraphDataset(train_pkl, train_emb)
    val_dataset   = PreprocessedGraphDataset(val_pkl, val_emb)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GEncoder(text_emb_dim=1152).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    print("\n=== Training Configuration ===")
    print(f"Device: {cfg.device}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Temperature: {cfg.temperature}")
    print("==============================\n")

    best_val_loss = float('inf')

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch_graph, batch_emb in pbar:
            batch_graph = batch_graph.to(cfg.device)
            batch_emb = batch_emb.to(cfg.device)
            
            optimizer.zero_grad()

            z_graph, _, _ = model(batch_graph)
            z_text = model.project_text(batch_emb)

            loss = contrastive_loss_improved(z_graph, z_text, temperature=cfg.temperature)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_graph, v_emb in val_loader:
                v_graph, v_emb = v_graph.to(cfg.device), v_emb.to(cfg.device)
                zg, _, _ = model(v_graph)
                zt = model.project_text(v_emb)
                v_l = contrastive_loss_improved(zg, zt, temperature=cfg.temperature)
                val_loss += v_l.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(cfg.base_path, cfg.model_name))
            print("Model saved (improved validation loss)")

    print("\nTraining complete!")

if __name__ == "__main__":
    train()