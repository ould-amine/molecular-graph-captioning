"""
Inference Script
"""

import os
import sys

# Add MolCA repo to path
MOLCA_PATH = os.environ.get('MOLCA_PATH', '.')
if MOLCA_PATH not in sys.path:
    sys.path.insert(0, MOLCA_PATH)

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm
import pandas as pd

# MolCA imports
try:
    from model.blip2_stage2 import Blip2Stage2
    from model.blip2_opt import smiles2data
    from data_provider.stage2_dm import smiles_handler

    HAS_MOLCA = True
except ImportError as e:
    print(f"Error importing MolCA modules: {e}")
    print("\nPlease ensure you're running from the MolCA repository directory,")
    print("or set MOLCA_PATH environment variable to the MolCA repo path.")
    HAS_MOLCA = False


def load_smiles_file(path: str) -> List[Tuple[str, str]]:
    """Loads SMILES from various file formats."""
    p = Path(path)
    pairs = []

    if p.suffix in ('.tsv', '.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')

            # Find column indices
            id_col = 'graph_id' if 'graph_id' in header else None
            smi_col = 'smiles' if 'smiles' in header else None

            if id_col is None:
                id_idx = 0
            else:
                id_idx = header.index(id_col)

            if smi_col is None:
                # Try to find SMILES column by position or content
                smi_idx = 1 if len(header) > 1 else 0
            else:
                smi_idx = header.index(smi_col)

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > max(id_idx, smi_idx):
                    gid = parts[id_idx]
                    smi = parts[smi_idx]
                    if smi:
                        pairs.append((gid, smi))

    elif p.suffix == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, list):
            for item in data:
                gid = getattr(item, 'id', None) or getattr(item, 'CID', None)
                smi = getattr(item, 'smiles', None) or getattr(item, 'SMILES', None)
                if gid and smi:
                    pairs.append((str(gid), smi))
        elif isinstance(data, dict):
            for gid, info in data.items():
                smi = info.get('smiles') or info.get('SMILES') if isinstance(info, dict) else None
                if smi:
                    pairs.append((str(gid), smi))
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    return pairs


def load_references(path: Optional[str]) -> Optional[Dict[str, str]]:
    """Loads reference descriptions for evaluation."""
    if path is None:
        return None

    refs = {}
    p = Path(path)

    if p.suffix == '.pkl':
        print("Loading reference descriptions from pickle...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, list):
            for item in data:
                gid = getattr(item, 'id', None)
                desc = getattr(item, 'description', None)
                if gid and desc:
                    refs[str(gid)] = desc
    elif p.suffix in ('.tsv', '.txt'):
        print("Loading reference descriptions from TSV/TXT...")
        with open(path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            id_idx = header.index('graph_id') if 'graph_id' in header else 0
            desc_idx = header.index('description') if 'description' in header else -1

            if desc_idx >= 0:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > max(id_idx, desc_idx):
                        refs[parts[id_idx]] = parts[desc_idx]
    print("Loaded reference descriptions:", len(refs))
    return refs if refs else None


def load_graphs_from_pkl(path: str) -> Dict[str, 'torch_geometric.data.Data']:
    """
    Loads pre-computed graphs from pickle file.

    Args:
        path: Path to .pkl file containing list of PyG Data objects

    Returns:
        Dictionary mapping graph ID to Data object
    """
    print(f"Loading pre-computed graphs from: {path}")
    with open(path, 'rb') as f:
        graphs = pickle.load(f)

    graph_dict = {}
    for g in graphs:
        gid = str(g.id)
        graph_dict[gid] = g

    print(f"Loaded {len(graph_dict)} pre-computed graphs")
    return graph_dict


class MolCAInferenceWrapper:
    """
    Wrapper for MolCA model inference.

    This class handles:
    1. Loading the full MolCA model from checkpoint
    2. Loading LoRA adapter weights for Galactica
    3. Converting SMILES to graphs
    4. Batched generation with proper prompt formatting
    """

    def __init__(
            self,
            checkpoint_path: str,
            peft_dir: str = None,
            device: str = 'cuda',
            opt_model: str = 'facebook/galactica-1.3b',
            gnn_ckpt: str = 'gin_pretrained/graphcl_80.pth',
    ):
        """
        Initialize MolCA model for inference.

        Args:
            checkpoint_path: Path to chebi.ckpt (contains GIN + Q-Former + Projection)
            peft_dir: Path to LoRA adapter directory (e.g., chebi_lora/) containing
                      adapter_config.json and adapter_model.bin
            device: Device to run inference on
            opt_model: Base language model name
            gnn_ckpt: Path to pre-trained GNN checkpoint
        """
        self.device = device
        self.collater = Collater([], [])

        # Create args namespace for model initialization
        # These should match the training configuration
        class Args:
            def __init__(self, peft_dir, opt_model, gnn_ckpt):
                self.bert_name = 'scibert'
                self.gin_num_layers = 5
                self.gin_hidden_dim = 300
                self.drop_ratio = 0.0
                self.tune_gnn = True
                self.num_query_token = 8
                self.cross_attention_freq = 2
                self.llm_tune = 'lora'
                self.peft_dir = peft_dir if peft_dir else ''
                self.opt_model = opt_model
                self.gnn_ckpt = gnn_ckpt
                self.prompt = '[START_I_SMILES]{}[END_I_SMILES]. '
                self.max_len = 384
                self.num_beams = 5
                self.do_sample = False
                self.min_len = 0
                self.reaction_weight = 1.0
                self.optimizer = 'adamw'
                self.caption_eval_epoch = 10

        args = Args(peft_dir, opt_model, gnn_ckpt)

        # Load model from checkpoint
        print(f"Loading MolCA from: {checkpoint_path}")
        if peft_dir:
            print(f"Loading LoRA adapter from: {peft_dir}")
        else:
            print("WARNING: No peft_dir specified! LoRA weights will NOT be loaded.")
            print("         For CheBI-20, use --peft_dir path/to/chebi_lora/")

        # Convert Args instance to dict for PyTorch Lightning compatibility
        args_dict = vars(args)

        model = Blip2Stage2.load_from_checkpoint(
            checkpoint_path,
            strict=False,
            args=args_dict
        )

        # Extract the core model (Blip2OPT)
        self.model = model.blip2opt
        print("Extracted Blip2OPT model:")
        print(self.model)
        print("Total parameters:", sum(p.numel() for p in self.model.parameters()))
        print("Trainable parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        # print("Total GNN (GIN) parameters:", sum(p.numel() for p in self.model.graph_encoder.parameters()))
        # print("Trainable GNN parameters:", sum(p.numel() for p in self.model.graph_encoder.parameters() if p.requires_grad))
        # print("Total Q-Former parameters:", sum(p.numel() for p in self.model.Qformer.parameters()))
        # print("Trainable Q-Former parameters:", sum(p.numel() for p in self.model.Qformer.parameters() if p.requires_grad))
        # non_lora_lm_params = [p for n, p in self.model.opt_model.named_parameters() if "lora" not in n]
        # print("Total Language Model parameters:", sum(p.numel() for p in non_lora_lm_params))
        # print("Trainable Language Model parameters:", sum(p.numel() for p in non_lora_lm_params if p.requires_grad))
        # print("Total OPT projection parameters:", sum(p.numel() for p in self.model.opt_proj.parameters()))
        # print("Trainable OPT projection parameters:", sum(p.numel() for p in self.model.opt_proj.parameters() if p.requires_grad))
        # import sys
        # sys.exit()
        del model

        # Move to device and set to eval mode
        self.model = self.model.half().eval().to(device)
        print("Model loaded successfully!")

    @torch.no_grad()
    def generate_caption(
            self,
            smiles: str,
            prompt_suffix: str = '',
            max_length: int = 256,
            num_beams: int = 5,
            temperature: float = 1.0,
            do_sample: bool = False,
    ) -> str:
        """Generates caption for a single SMILES string."""

        # Convert SMILES to graph
        try:
            graph_batch = self.collater([smiles2data(smiles)]).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to convert SMILES to graph: {e}")

        # Prepare prompt with SMILES and molecule placeholders
        prompt = f'[START_I_SMILES]{smiles[:256]}[END_I_SMILES]. {prompt_suffix}'
        prompt = smiles_handler(prompt, '<mol>' * 8, True)[0]

        # Tokenize
        self.model.opt_tokenizer.padding_side = 'left'
        prompt_batch = self.model.opt_tokenizer(
            [prompt],
            truncation=True,
            padding='longest',
            add_special_tokens=True,
            max_length=384,
            return_tensors='pt',
            return_attention_mask=True
        ).to(self.device)

        # Mark molecule token positions
        is_mol_token = prompt_batch.input_ids == self.model.opt_tokenizer.mol_token_id
        prompt_batch['is_mol_token'] = is_mol_token

        # Prepare samples dict
        samples = {
            'graphs': graph_batch,
            'prompt_tokens': prompt_batch
        }

        # Generate
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            text = self.model.generate(
                samples,
                temperature=temperature,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample
            )[0]

        return text

    @torch.no_grad()
    def generate_batch(
            self,
            smiles_list: List[str],
            max_length: int = 256,
            num_beams: int = 5,
            temperature: float = 1.0,
            do_sample: bool = False,
            precomputed_graphs: List = None
    ) -> List[str]:
        """Generates captions for a batch of SMILES strings."""

        # Convert all SMILES to graphs
        graphs = []
        valid_indices = []

        if precomputed_graphs is not None:
            # Use pre-computed graphs directly
            for i, graph in enumerate(precomputed_graphs):
                if graph is not None:
                    graphs.append(graph)
                    valid_indices.append(i)
                else:
                    print(f"Warning: No pre-computed graph at index {i}")
        else:

            for i, smi in enumerate(smiles_list):
                try:
                    graphs.append(smiles2data(smi))
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Warning: Skipping invalid SMILES at index {i}: {e}")

        if not graphs:
            return [''] * len(smiles_list)

        graph_batch = self.collater(graphs).to(self.device)

        # Prepare prompts
        prompts = []
        for i in valid_indices:
            smi = smiles_list[i]
            prompt = f'[START_I_SMILES]{smi[:256]}[END_I_SMILES]. '
            prompt = smiles_handler(prompt, '<mol>' * 8, True)[0]
            prompts.append(prompt)

        # Tokenize
        self.model.opt_tokenizer.padding_side = 'left'
        prompt_batch = self.model.opt_tokenizer(
            prompts,
            truncation=True,
            padding='longest',
            add_special_tokens=True,
            max_length=384,
            return_tensors='pt',
            return_attention_mask=True
        ).to(self.device)

        is_mol_token = prompt_batch.input_ids == self.model.opt_tokenizer.mol_token_id
        prompt_batch['is_mol_token'] = is_mol_token

        samples = {
            'graphs': graph_batch,
            'prompt_tokens': prompt_batch
        }

        # Generate
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            texts = self.model.generate(
                samples,
                temperature=temperature,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample
            )

        # Map back to original indices
        results = [''] * len(smiles_list)
        for i, text in zip(valid_indices, texts):
            results[i] = text

        return results


def load_trainable_delta(model, delta_ckpt_path: str, device: str = "cpu"):
    """
    Loads a trainable-only delta checkpoint (saved by save_checkpoint) and applies it
    onto an already-initialized base model using strict=False.
    """
    ckpt = torch.load(delta_ckpt_path, map_location="cpu")
    delta_state = ckpt.get("state_dict", ckpt)

    remapped = {}
    for k, v in delta_state.items():
        if k.startswith("blip2opt."):
            remapped[k[len("blip2opt."):]] = v
        else:
            remapped[k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    model.to(device)

    print(f"[Delta] Applied trainable-only checkpoint to Blip2OPT: {delta_ckpt_path}")
    print(f"[Delta] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0:
        print(f"[Delta] Unexpected (first 10): {unexpected[:10]}")
    return ckpt


def main():
    parser = argparse.ArgumentParser(description='MolCA Molecule Captioning')
    parser.add_argument('--checkpoint', required=True, help='Path to MolCA checkpoint (e.g., chebi.ckpt)')
    parser.add_argument('--peft_dir', required=True, help='Path to LoRA adapter directory (e.g., chebi_lora/)')
    parser.add_argument('--graph_pkl', default=None,
                        help='Path to pre-computed graphs pickle file. '
                             'If provided, graphs are loaded directly instead of '
                             'converting from SMILES.')
    parser.add_argument('--smiles_file', required=True, help='Input SMILES file')
    parser.add_argument('--output_csv', default='submission.csv', help='Output CSV path')
    parser.add_argument('--refs_file', default=None, help='Reference file for evaluation')
    parser.add_argument('--opt_model', default='facebook/galactica-1.3b')
    parser.add_argument('--gnn_ckpt', default='gin_pretrained/graphcl_80.pth', help='Path to GNN checkpoint')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='Limit samples (0=all)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--trainable_params_only', action='store_true', default=False,
                        help='If set, args.checkpoint is treated as a trainable-only delta checkpoint and '
                             'the base weights are loaded from --base_checkpoint.')

    parser.add_argument('--base_checkpoint', type=str, default='',
                        help='Path to the FULL base checkpoint (e.g., chebi.ckpt). Required if --trainable_params_only is set.')

    args = parser.parse_args()

    if not HAS_MOLCA:
        print("Cannot proceed without MolCA modules.")
        sys.exit(1)

    # Load data
    print(f"\nLoading SMILES from: {args.smiles_file}")
    pairs = load_smiles_file(args.smiles_file)
    if args.limit > 0:
        pairs = pairs[:args.limit]
    print(f"Loaded {len(pairs)} molecules")

    refs = load_references(args.refs_file)

    # Load pre-computed graphs if provided
    precomputed_graph_dict = None
    if args.graph_pkl:
        precomputed_graph_dict = load_graphs_from_pkl(args.graph_pkl)

    if args.trainable_params_only:
        if not args.base_checkpoint:
            raise ValueError("--base_checkpoint is required when --trainable_params_only is set.")

        model = MolCAInferenceWrapper(
            checkpoint_path=args.base_checkpoint,
            peft_dir=args.peft_dir,
            device=args.device,
            opt_model=args.opt_model,
            gnn_ckpt=args.gnn_ckpt
        )

        target_module = model.model if hasattr(model, "model") else model
        load_trainable_delta(target_module, args.checkpoint, device=args.device)

    else:
        # Default: args.checkpoint is a full checkpoint
        model = MolCAInferenceWrapper(
            checkpoint_path=args.checkpoint,
            peft_dir=args.peft_dir,
            device=args.device,
            opt_model=args.opt_model,
            gnn_ckpt=args.gnn_ckpt
        )

    # Generate captions
    print(f"\nGenerating captions (batch_size={args.batch_size})...")

    all_ids = []
    all_preds = []

    for batch_start in tqdm(range(0, len(pairs), args.batch_size)):
        batch = pairs[batch_start:batch_start + args.batch_size]
        batch_ids = [p[0] for p in batch]
        batch_smiles = [p[1] for p in batch]

        # Get pre-computed graphs for this batch if available
        batch_graphs = None
        if precomputed_graph_dict is not None:
            batch_graphs = [precomputed_graph_dict.get(gid) for gid in batch_ids]
            missing = sum(1 for g in batch_graphs if g is None)
            if missing > 0:
                print(f"Warning: {missing}/{len(batch_ids)} graphs not found in pre-computed dict")

        captions = model.generate_batch(
            batch_smiles,
            max_length=args.max_length,
            num_beams=args.num_beams,
            temperature=args.temperature,
            do_sample=args.do_sample,
            precomputed_graphs=batch_graphs
        )

        all_ids.extend(batch_ids)
        all_preds.extend(captions)

    # Save results
    df = pd.DataFrame({'ID': all_ids, 'description': all_preds})
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(df)} predictions to: {args.output_csv}")

    # Evaluate if references available
    if refs:
        print("\n" + "=" * 50)
        print("Evaluation")
        print("=" * 50)

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smooth = SmoothingFunction().method1

            bleu_scores = []
            for gid, pred in zip(all_ids, all_preds):
                ref = refs.get(gid)
                if ref and pred:
                    score = sentence_bleu(
                        [ref.split()], pred.split(),
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smooth
                    )
                    bleu_scores.append(score)

            if bleu_scores:
                print(f"BLEU-4: {sum(bleu_scores) / len(bleu_scores):.4f}")
        except Exception as e:
            print(f"BLEU computation skipped: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
