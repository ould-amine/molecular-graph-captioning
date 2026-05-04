import os
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

BASE_PATH = "."
MODEL_A = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_B = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("Loading SentenceBERT models...")
model_a = SentenceTransformer(MODEL_A, device=DEVICE)
model_b = SentenceTransformer(MODEL_B, device=DEVICE)
print("Models loaded.")

def enrich_description(graph):
    """Ajoute des informations structurelles à la description texte."""
    desc = graph.description
    extras = []
    if graph.num_nodes > 50:
        extras.append("large molecule")
    else:
        extras.append("small molecule")
        
    if graph.edge_index.size(1) / graph.num_nodes > 1.5:
        extras.append("dense bonds")
        
    extra = " [Molecule info] " + " ".join(extras)
    return desc + extra

def generate_text_embeddings(split="train"):
    """Génère et sauvegarde les embeddings concaténés MiniLM + MPNet."""
    pkl_path = os.path.join(BASE_PATH, "data", f"{split}_graphs.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"Erreur : Le fichier {pkl_path} est introuvable.")
        return

    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs for {split}")

    descriptions = [enrich_description(g) for g in graphs]
    ids = [g.id for g in graphs]

    print(f"Encoding {split} with MiniLM...")
    emb_a = model_a.encode(descriptions, batch_size=64, convert_to_numpy=True, 
                           normalize_embeddings=True, show_progress_bar=True)

    print(f"Encoding {split} with MPNet...")
    emb_b = model_b.encode(descriptions, batch_size=32, convert_to_numpy=True, 
                           normalize_embeddings=True, show_progress_bar=True)
    emb_a = torch.from_numpy(emb_a)
    emb_b = torch.from_numpy(emb_b)
    embeddings = torch.cat([emb_a, emb_b], dim=1)
    embeddings = F.normalize(embeddings, dim=-1).numpy()

   
    df = pd.DataFrame({
        "ID": ids,
        "embedding": [",".join(map(str, e)) for e in embeddings]
    })
    
    output_dir = os.path.join(BASE_PATH, "data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, f"{split}_embeddings.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved embeddings to {output_path}")

if __name__ == "__main__":
    generate_text_embeddings("train")
    generate_text_embeddings("validation")