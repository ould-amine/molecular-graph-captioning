"""
MolCA Fine-Tuning Script
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from rdkit import RDLogger
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Batch
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

# Add MolCA repo to path
MOLCA_PATH = os.environ.get('MOLCA_PATH', '.')
if MOLCA_PATH not in sys.path:
    sys.path.insert(0, MOLCA_PATH)

# MolCA imports
try:
    from model.blip2_stage2 import Blip2Stage2
    from model.blip2_opt import smiles2data, smiles_handler, escape_custom_split_sequence
    from model.help_funcs import AttrDict

    HAS_MOLCA = True
except ImportError as e:
    print(f"Error importing MolCA modules: {e}")
    print("\nPlease ensure you're running from the MolCA repository directory,")
    print("or set MOLCA_PATH environment variable to the MolCA repo path.")
    HAS_MOLCA = False
    sys.exit(1)

# Evaluation imports
try:
    from nltk.translate.bleu_score import corpus_bleu
    from rouge_score import rouge_scorer

    HAS_EVAL = True
except ImportError:
    print("Warning: nltk or rouge_score not installed. Evaluation will be limited.")
    HAS_EVAL = False

try:
    from bert_score import score as bertscore_score

    HAS_BERTSCORE = True
except ImportError:
    print("Warning: bert_score not installed. BERTScore will be disabled.")
    HAS_BERTSCORE = False


# ===============
# Data Loading
# ===============

def load_smiles_file(path: str) -> List[Tuple[str, str, str]]:
    """
    Loads SMILES and descriptions from TSV file.
    
    Expected format (TSV with header):
        graph_id    description     smiles
    Returns:
        List of (id, description, smiles) tuples
    """
    data = []
    p = Path(path)

    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')

        # Find column indices
        id_col = 'graph_id' if 'graph_id' in header else None
        desc_col = 'description' if 'description' in header else None
        smi_col = 'smiles' if 'smiles' in header else None

        id_idx = header.index(id_col) if id_col else 0
        desc_idx = header.index(desc_col) if desc_col else 1
        smi_idx = header.index(smi_col) if smi_col else 2

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > max(id_idx, smi_idx, desc_idx):
                gid = parts[id_idx]
                desc = parts[desc_idx]
                smi = parts[smi_idx]
                if smi:
                    data.append((gid, desc, smi))

    return data


class MoleculeDataset(Dataset):
    """
    Dataset for molecule captioning training.
    
    Each item contains:
        - graph: PyG Data object from SMILES
        - description: Ground truth text
        - smiles: SMILES string (for prompt)
    """

    def __init__(
            self,
            data: List[Tuple[str, str, str]],
            max_smiles_len: int = 256,
            skip_invalid: bool = False
    ):
        """
        Args:
            data: List of (id, description, smiles) tuples
            max_smiles_len: Maximum SMILES length to include in prompt
            skip_invalid: Skip molecules that fail SMILES->graph conversion
        """
        self.data = []
        self.max_smiles_len = max_smiles_len

        print(f"Converting {len(data)} SMILES to graphs...")
        # failed = 0
        for gid, desc, smiles in tqdm(data, desc="Processing SMILES"):
            try:
                graph = smiles2data(smiles)
                graph.id = gid
                self.data.append({
                    'id': gid,
                    'graph': graph,
                    'description': desc,
                    'smiles': smiles  # [:max_smiles_len],
                })
            except Exception as e:
                # failed += 1
                if not skip_invalid:
                    raise e

        print(f"Dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_train(batch, tokenizer, prompt_template, num_query_tokens, text_max_len=256):
    """
    Collate function for training batches.
    
    Prepares:
        - graphs: Batched PyG graphs
        - prompt_tokens: Tokenized prompts with molecule placeholders
        - text_tokens: Tokenized descriptions (targets)
    
    Args:
        batch: List of dataset items
        tokenizer: Galactica tokenizer
        prompt_template: Template like '[START_I_SMILES]{}[END_I_SMILES]. '
        num_query_tokens: Number of Q-Former query tokens
        text_max_len: Maximum length for text tokenization
    """
    graphs = []
    prompts = []
    descriptions = []

    mol_placeholder = '<mol>' * num_query_tokens

    for item in batch:
        graphs.append(item['graph'])

        # Truncate SMILES to prevent prompt from being too long
        smiles = item['smiles'][:300]  # We truncate SMILES to 300 chars max
        # Format prompt with SMILES
        prompt = prompt_template.format(smiles)
        prompt, _ = smiles_handler(prompt, mol_placeholder)
        prompts.append(prompt)

        descriptions.append(item['description'])

    graph_batch = Batch.from_data_list(graphs)

    # Tokenize prompts (right padding for training)
    tokenizer.padding_side = 'right'
    prompt_tokens = tokenizer(
        prompts,
        truncation=False,
        padding='longest',
        add_special_tokens=True,
        # max_length=384,
        return_tensors='pt',
        return_attention_mask=True
    )

    # Mark molecule token positions
    is_mol_token = prompt_tokens.input_ids == tokenizer.mol_token_id
    prompt_tokens['is_mol_token'] = is_mol_token

    # Tokenize descriptions (targets)
    text_tokens = tokenizer(
        descriptions,
        truncation=True,
        padding='longest',
        add_special_tokens=True,
        max_length=text_max_len,
        return_tensors='pt',
        return_attention_mask=True
    )

    return graph_batch, prompt_tokens, text_tokens


def collate_fn_eval(batch, tokenizer, prompt_template, num_query_tokens):
    """
    Collate function for evaluation batches (no text tokenization needed).
    """
    graphs = []
    prompts = []
    descriptions = []
    ids = []

    mol_placeholder = '<mol>' * num_query_tokens

    for item in batch:
        graphs.append(item['graph'])
        ids.append(item['id'])

        smiles = item['smiles'][:300]
        prompt = prompt_template.format(smiles)
        prompt, _ = smiles_handler(prompt, mol_placeholder, )
        prompts.append(prompt)

        descriptions.append(item['description'])

    graph_batch = Batch.from_data_list(graphs)

    tokenizer.padding_side = 'left'  # Left padding for generation
    prompt_tokens = tokenizer(
        prompts,
        truncation=False,
        padding='longest',
        add_special_tokens=True,
        # max_length=384,
        return_tensors='pt',
        return_attention_mask=True
    )

    is_mol_token = prompt_tokens.input_ids == tokenizer.mol_token_id
    prompt_tokens['is_mol_token'] = is_mol_token

    return graph_batch, prompt_tokens, descriptions, ids


# =============
# Evaluation
# =============

def compute_bleu4(predictions: List[str], references: List[str]) -> float:
    """
    Computes BLEU-4 score.

    Args:
        predictions: List of generated captions
        references: List of ground truth captions

    Returns:
        BLEU-4 score (0-100)
    """
    if not HAS_EVAL:
        return 0.0

    hyps = []
    refs = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        hyps.append(pred_tokens)
        refs.append([ref_tokens])

    bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu4 * 100


def compute_bertscore_f1(predictions: List[str], references: List[str],
                         lang: str = "en", model_type: str = "roberta-base",
                         device: str = "cuda") -> float:
    """
    Computes BERTScore F1 (0-100).
    """
    if not HAS_BERTSCORE:
        return 0.0

    P, R, F1 = bertscore_score(
        predictions,
        references,
        lang=lang,
        model_type=model_type,
        device=device,
        verbose=False
    )
    return float(F1.mean().item() * 100.0)


@torch.no_grad()
def evaluate(model, dataloader, device, num_beams=5, max_length=256):
    """
    Evaluates a model on the validation set.
    
    Returns:
        Dictionary with metrics and predictions
    """
    model.eval()

    all_predictions = []
    all_references = []
    all_ids = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        graphs, prompt_tokens, descriptions, ids = batch

        graphs = graphs.to(device)
        prompt_tokens = dict_to_namespace(prompt_tokens, device)

        samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = model.blip2opt.generate(
                samples,
                num_beams=num_beams,
                max_length=max_length,
                min_length=1,
                do_sample=False
            )

        all_predictions.extend(predictions)
        all_references.extend(descriptions)
        all_ids.extend(ids)

    # Compute BLEU-4
    bleu4 = compute_bleu4(all_predictions, all_references)

    # Compute BERTScore
    bertscore_f1 = compute_bertscore_f1(
        all_predictions, all_references,
        model_type="roberta-base",
        device=str(device)
    )

    # Average metric (0-100 scale)
    avg_metric = 0.5 * (bleu4 + bertscore_f1)

    model.train()

    return {
        'bleu4': bleu4,
        'bertscore_f1': bertscore_f1,
        'avg_metric': avg_metric,
        'predictions': all_predictions,
        'references': all_references,
        'ids': all_ids
    }


# ==========
# Training
# ==========

class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-1, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics where higher is better (BLEU), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if we should stop.
        
        Returns:
            True if this is a new best score
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True

        if self.mode == 'max':
            improved = score >= self.best_score + self.min_delta
        else:
            improved = score <= self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path, pytorch_lightning_version='1.9.0'):
    """Saves training checkpoint."""
    # Get trainable state dict
    state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.data.cpu()

    checkpoint = {
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'pytorch-lightning_version': pytorch_lightning_version
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def save_lora_adapter(model, output_dir):
    """Saves LoRA adapter separately."""
    lora_path = Path(output_dir) / 'lora_adapter'
    lora_path.mkdir(parents=True, exist_ok=True)

    # Save using PEFT's method
    model.blip2opt.opt_model.save_pretrained(str(lora_path))
    print(f"Saved LoRA adapter to {lora_path}")


def dict_to_namespace(d, device=None):
    """Converts dict to SimpleNamespace for attribute access, optionally moving tensors to device."""
    data = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and device is not None:
            data[k] = v.to(device)
        else:
            data[k] = v
    return SimpleNamespace(**data)


def freeze_some_layers(model: nn.Module):
    """
    Freezes everything, then unfreezes specific layers
    """
    # Freeze all params first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze LoRA params by name
    # num_lora = 0
    # for name, p in model.named_parameters():
    #     if "lora_" in name:
    #         p.requires_grad = True
    #         num_lora += p.numel()

    # Unfreeze cross-modal projection params
    num_proj = 0
    for name, p in model.named_parameters():
        if "opt_proj" in name:
            p.requires_grad = True
            num_proj += p.numel()

    # print(f"[Freeze] Unfroze LoRA params: {num_lora:,} | cross-modal projection params: {num_proj:,}")
    print(f"[Freeze] Unfroze cross-modal projection params: {num_proj}")


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, grad_accum_steps=1, global_step=0,
                warmup_steps=0, quick_eval_interval_steps=0, quick_eval_fn=None):
    """
    Trains for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_opt_step = 0  # number of optimizer steps taken in this epoch

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        graphs, prompt_tokens, text_tokens = batch

        # Move to device
        graphs = graphs.to(device)
        prompt_tokens = dict_to_namespace(prompt_tokens, device)
        text_tokens = dict_to_namespace(text_tokens, device)

        # Forward pass with mixed precision
        with autocast(dtype=torch.bfloat16):
            loss_dict = model.blip2opt((graphs, prompt_tokens, text_tokens))
            loss = loss_dict['loss'] / grad_accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:

            if warmup_steps > 0 and global_step < warmup_steps:
                warmup_update_lr(optimizer, global_step, warmup_steps)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Step scheduler only AFTER warmup, and reset its "time" to start at 0 after warmup
            if scheduler is not None and global_step >= warmup_steps:
                scheduler.step(global_step - warmup_steps + 1)

            global_step += 1
            epoch_opt_step += 1

            # Quick eval every X fraction of an epoch
            if (quick_eval_fn is not None and quick_eval_interval_steps > 0
                    and (epoch_opt_step % quick_eval_interval_steps == 0)):
                quick_eval_fn(epoch_opt_step=epoch_opt_step, global_step=global_step)

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item() * grad_accum_steps:.4f}",
            'lr_proj': f"{optimizer.param_groups[0]['lr']:.2e}"
            # 'lr_lora': f"{optimizer.param_groups[1]['lr']:.2e}",
        })

    return total_loss / num_batches, global_step


def warmup_update_lr(optimizer, step: int, warmup_steps: int):
    """
    Linearly warms up each param group's LR from 0 to base_lr over warmup_steps optimizer steps.
    step is 0-indexed global optimizer step.
    """
    if step >= warmup_steps:
        return
    scale = float(step + 1) / float(warmup_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = pg["initial_lr"] * scale


def make_sequential_quick_val_loader_by_counter(
        val_dataset,
        val_collate,
        eval_batch_size: int,
        num_workers: int,
        fraction: float,
        counter: int
):
    """
    Returns a DataLoader over a sequential (contiguous) slice of val_dataset.

    Uses chunk size k = int(fraction * N). The slice advances each epoch:
      counter 0 -> [0:k)
      counter 1 -> [k:2k)
      ...
    Wraps around when reaching the end.
    """
    n = len(val_dataset)
    k = max(1, int(n * fraction))

    start = (counter * k) % n
    end = start + k

    if end <= n:
        indices = list(range(start, end))
    else:
        # wrap around
        indices = list(range(start, n)) + list(range(0, end - n))

    subset = Subset(val_dataset, indices)

    return DataLoader(
        subset,
        batch_size=eval_batch_size,
        shuffle=False,  # keep sequential order
        num_workers=num_workers,
        collate_fn=val_collate,
        pin_memory=True
    ), (start, min(end, n), (end - n) if end > n else None)


def main():
    parser = argparse.ArgumentParser(description='MolCA Fine-Tuning')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to chebi.ckpt checkpoint')
    parser.add_argument('--peft_dir', type=str, required=True,
                        help='Path to LoRA adapter directory (e.g., chebi_lora/)')
    parser.add_argument('--gnn_ckpt', type=str, default='gin_pretrained/graphcl_80.pth',
                        help='Path to GIN pretrained weights')
    parser.add_argument('--opt_model', type=str, default='facebook/galactica-1.3b',
                        help='Base language model')

    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training TSV file')
    parser.add_argument('--val_file', type=str, required=True,
                        help='Path to validation TSV file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/finetuned',
                        help='Output directory for checkpoints')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Evaluation batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Number of warmup epochs')

    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')

    parser.add_argument('--tune_gnn', action='store_true', default=False,
                        help='Fine-tune GNN encoder')
    parser.add_argument('--num_query_token', type=int, default=8,
                        help='Number of Q-Former query tokens')
    parser.add_argument('--text_max_len', type=int, default=384,
                        help='Maximum text length')
    parser.add_argument('--num_beams', type=int, default=5,
                        help='Beam size for generation')
    parser.add_argument('--prompt', type=str, default='[START_I_SMILES]{}[END_I_SMILES]. ',
                        help='Prompt template')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--val_percentage', type=float, default=100.0,
                        help='Percentage of validation set to use (0-100)')

    parser.add_argument('--lr_lora', type=float, default=2e-4,
                        help='Learning rate for LoRA params')
    parser.add_argument('--lr_proj', type=float, default=5e-5,
                        help='Learning rate for projection params')
    parser.add_argument('--wd_lora', type=float, default=0.0,
                        help='Weight decay for LoRA params')
    parser.add_argument('--wd_proj', type=float, default=0.01,
                        help='Weight decay for projection params')
    parser.add_argument('--t0_epochs', type=int, default=10,
                        help='Cosine warm restarts: first cycle length in epochs (T_0)')
    parser.add_argument('--t_mult', type=int, default=2,
                        help='Cosine warm restarts: cycle length multiplier (T_mult)')
    parser.add_argument('--quick_val_fraction', type=float, default=0.10,
                        help='Fraction of validation set used for quick per-epoch logging (0-1).')
    parser.add_argument('--quick_eval_every', type=float, default=1.0,
                        help='Run quick evaluation every this fraction of an epoch (e.g., 0.25 -> 4 times/epoch).')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("MolCA Fine-Tuning")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"LoRA adapter: {args.peft_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # =============
    # Load Model
    # =============
    print("\nLoading model...")

    model_args = AttrDict(
        bert_name='scibert',
        gin_num_layers=5,
        gin_hidden_dim=300,
        drop_ratio=0.0,
        tune_gnn=args.tune_gnn,
        num_query_token=args.num_query_token,
        cross_attention_freq=2,
        llm_tune='lora',
        peft_dir=args.peft_dir,
        opt_model=args.opt_model,
        gnn_ckpt=args.gnn_ckpt,
        prompt=args.prompt,
        max_len=args.text_max_len,
        min_len=1,
        num_beams=args.num_beams,
        do_sample=False,
        reaction_weight=1.0,
        optimizer='adamw',
        caption_eval_epoch=1,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        peft_config=None,
        init_checkpoint='',
        stage1_path='',
        stage2_path='',
    )

    model = Blip2Stage2.load_from_checkpoint(
        args.checkpoint,
        strict=False,
        args=vars(model_args)
    )

    model = model.to(args.device)

    freeze_some_layers(model)

    tokenizer = model.blip2opt.opt_tokenizer

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # ===========
    # Load Data
    # ===========
    print("\nLoading data...")

    train_data = load_smiles_file(args.train_file)
    val_data = load_smiles_file(args.val_file)

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    train_dataset = MoleculeDataset(train_data)
    val_dataset = MoleculeDataset(val_data)

    # Subset validation dataset if val_percentage < 100
    if args.val_percentage < 100.0:
        val_size = int(len(val_dataset) * args.val_percentage / 100.0)
        val_size = max(1, val_size)  # Ensure at least 1 sample
        val_indices = list(range(val_size))
        val_dataset = Subset(val_dataset, val_indices)
        print(f"Using {args.val_percentage}% of validation set: {len(val_dataset)} samples")

    # Create data loaders with custom collate functions
    train_collate = lambda batch: collate_fn_train(
        batch, tokenizer, args.prompt, args.num_query_token, args.text_max_len
    )
    val_collate = lambda batch: collate_fn_eval(
        batch, tokenizer, args.prompt, args.num_query_token
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate,
        pin_memory=True,
        drop_last=True
    )

    val_loader_full = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_collate,
        pin_memory=True
    )

    # ================
    # Setup Training
    # ================
    print("\nSetting up training...")

    # Optimizer with discriminative LRs
    lora_params = []
    proj_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(p)
        else:
            proj_params.append(p)

    param_groups = []
    if len(lora_params) > 0:
        param_groups.append({
            "params": lora_params,
            "lr": 0.0,
            "initial_lr": args.lr_lora,  # what the scheduler will restart to
            "weight_decay": args.wd_lora
        })
    if len(proj_params) > 0:
        param_groups.append({
            "params": proj_params,
            "lr": 0.0,
            "initial_lr": args.lr_proj,
            "weight_decay": args.wd_proj
        })

    optimizer = AdamW(param_groups)
    print(f"[Opt] #LoRA tensors: {len(lora_params)} | #Proj tensors: {len(proj_params)}")

    # Cosine annealing with warm restarts scheduler
    # We'll handle warmup manually in the first few epochs
    # We account for gradient accumulation when computing total optimizer steps
    steps_per_epoch = len(train_loader) // args.grad_accum_steps

    # How often to quick-evaluate within an epoch (in optimizer steps)
    quick_eval_interval_steps = 0
    if args.quick_eval_every is not None and args.quick_eval_every > 0:
        quick_eval_interval_steps = max(1, int(steps_per_epoch * args.quick_eval_every))

    # Convert epoch-based restart lengths to step-based (CosineAnnealingWarmRestarts uses "iterations")
    t0_steps = args.t0_epochs * steps_per_epoch
    warmup_steps = steps_per_epoch * args.warmup_epochs
    # T_max is the number of optimizer steps after warmup
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=t0_steps,  # first restart after this many optimizer steps
        T_mult=args.t_mult,  # cycle length multiplier
        eta_min=args.min_lr
    )

    # Mixed precision scaler
    scaler = GradScaler()

    early_stopping = EarlyStopping(patience=args.patience, mode='max')

    # ================
    # Training Loop
    # ================
    print("\nStarting training...")
    print(f"  Epochs: {args.max_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum_steps}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f" Initial LR (LoRA): {args.lr_lora}")
    print(f" Initial LR (Proj): {args.lr_proj}")
    print(f"  Weight decay (LoRA): {args.wd_lora}")
    print(f"  Weight decay (Proj): {args.wd_proj}")
    print(f"  Warmup epochs: {args.warmup_epochs} ({warmup_steps} steps)")
    print(f"  Early stopping patience: {args.patience}")
    print("=" * 60)

    training_log = []

    best_avg = 0.0
    best_avg_bleu4 = 0.0
    best_avg_bertscore_f1 = 0.0
    global_step = 0
    quick_eval_counter = 0

    def quick_eval_callback(epoch_opt_step: int, global_step: int):
        nonlocal quick_eval_counter

        quick_loader, slice_info = make_sequential_quick_val_loader_by_counter(
            val_dataset=val_dataset,
            val_collate=val_collate,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            fraction=args.quick_val_fraction,
            counter=0  # quick_eval_counter
        )

        start, end_no_wrap, wrap_end = slice_info
        if wrap_end is None:
            print(
                f"  [QuickEval] step_in_epoch={epoch_opt_step} global_step={global_step} val slice [{start}:{end_no_wrap})")
        else:
            print(
                f"  [QuickEval] step_in_epoch={epoch_opt_step} global_step={global_step} val slices [{start}:{end_no_wrap}) + [0:{wrap_end})")

        results = evaluate(
            model, quick_loader, args.device,
            num_beams=args.num_beams, max_length=args.text_max_len
        )

        print(f"  [QuickEval] BLEU-4: {results['bleu4']:.2f} | "
              f"BERTScore-F1: {results.get('bertscore_f1', 0.0):.2f} | "
              f"Avg: {results.get('avg_metric', 0.0):.2f}")

        quick_eval_counter += 1

    for epoch in range(args.max_epochs):
        eval_results = None
        # quick_results = None
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")

        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer,
            scheduler, args.device, scaler,
            args.grad_accum_steps,
            global_step=global_step,
            warmup_steps=warmup_steps,
            quick_eval_interval_steps=quick_eval_interval_steps,
            quick_eval_fn=quick_eval_callback
        )

        print(f"  Train Loss: {train_loss:.4f}")

        # ---- Quick eval on sequential 10% chunk (every epoch) ----
        # quick_loader, slice_info = make_sequential_quick_val_loader(
        #     val_dataset=val_dataset,
        #     val_collate=val_collate,
        #     eval_batch_size=args.eval_batch_size,
        #     num_workers=args.num_workers,
        #     fraction=args.quick_val_fraction,
        #     epoch=epoch  # epoch is 0-indexed in your loop
        # )
        #
        # start, end_no_wrap, wrap_end = slice_info
        # if wrap_end is None:
        #     print(f"  Quick evaluating on val slice [{start}:{end_no_wrap})")
        # else:
        #     print(f"  Quick evaluating on val slices [{start}:{end_no_wrap}) + [0:{wrap_end})")
        #
        # quick_results = evaluate(
        #     model, quick_loader, args.device,
        #     num_beams=args.num_beams, max_length=args.text_max_len
        # )
        #
        # print(f"  Quick Val BLEU-4: {quick_results['bleu4']:.2f} | "
        #       f"BERTScore-F1: {quick_results.get('bertscore_f1', 0.0):.2f} | "
        #       f"Avg: {quick_results.get('avg_metric', 0.0):.2f}")

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            print("  Evaluating...")
            eval_results = evaluate(
                model, val_loader_full, args.device,
                num_beams=args.num_beams, max_length=args.text_max_len
            )

            bleu4 = eval_results['bleu4']
            bertscore_f1 = eval_results['bertscore_f1']
            avg_metric = eval_results['avg_metric']
            print(f"  Validation BLEU-4: {bleu4:.2f} | BERTScore-F1: {bertscore_f1:.2f} | Avg: {avg_metric:.2f}")

            # Log
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_bleu4': bleu4,
                'val_bertscore_f1': bertscore_f1,
                'val_avg_metric': avg_metric,
                'lr_proj': optimizer.param_groups[0]['lr'],
                # 'lr_lora': optimizer.param_groups[1]['lr'],
            }
            training_log.append(log_entry)

            # Save log
            with open(output_dir / 'training_log.json', 'w') as f:
                json.dump(training_log, f, indent=2)

            # Check for improvement
            is_best = early_stopping(avg_metric, epoch + 1)

            if is_best:
                best_avg = avg_metric
                best_avg_bleu4 = bleu4
                best_avg_bertscore_f1 = bertscore_f1
                print(f"  New best avg metric: {best_avg:.2f} at epoch {epoch + 1}")

                # Save best checkpoint
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    {'bleu4': bleu4, 'bertscore_f1': bertscore_f1, 'avg_metric': avg_metric, 'train_loss': train_loss},
                                                 output_dir / 'best_model.ckpt',
                )

                # Save LoRA adapter separately
                save_lora_adapter(model, output_dir)

                # Save predictions
                with open(output_dir / 'best_predictions.json', 'w') as f:
                    json.dump({
                        'predictions': eval_results['predictions'],
                        'references': eval_results['references'],
                        'ids': eval_results['ids']
                    }, f, indent=2)

            # Early stopping check
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best BLEU-4: {early_stopping.best_score:.2f} at epoch {early_stopping.best_epoch}")
                break

        # Save checkpoint
        # if (epoch + 1) % 10 == 0:
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            {
                "train_loss": train_loss,
                # "quick_val_bleu4": quick_results.get("bleu4", None),
                # "quick_val_bertscore_f1": quick_results.get("bertscore_f1",
                #                                            None),
                # "quick_val_avg_metric": quick_results.get("avg_metric", None),
                "full_val_bleu4": eval_results.get("bleu4", None) if eval_results else None,
                "full_val_bertscore_f1": eval_results.get("bertscore_f1", None) if eval_results else None,
                "full_val_avg_metric": eval_results.get("avg_metric", None) if eval_results else None,
                "global_step": global_step,
            },
                                         output_dir / "latest_model.ckpt"
        )

    # ================
    # Final Summary
    # ================
    print("\n" + "=" * 60)
    print("Training Complete.")
    print("=" * 60)
    print(f"Best Avg Metric: {best_avg:.2f}")
    print(f"Best BLEU-4: {best_avg_bleu4:.2f}")
    print(f"Best BERTScore-F1: {best_avg_bertscore_f1:.2f}")
    print(f"Best epoch: {early_stopping.best_epoch}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"  - best_model.ckpt")
    print(f"  - lora_adapter/")
    print("=" * 60)


if __name__ == '__main__':
    main()
