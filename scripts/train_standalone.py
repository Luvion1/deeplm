"""
Standalone training script — works anywhere (Docker, Modal, bare metal).

Usage:
    # Local
    python scripts/train_standalone.py --num-rows 50000 --max-steps 5000
    
    # Docker
    docker build -t deeplm-train .
    docker run --gpus all deeplm-train --num-rows 50000 --max-steps 5000
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeplm.config import DeeplmConfig
from deeplm.model.deeplm import DeeplmModel
from deeplm.data.kbi_dataset import MappedKBBIDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")

    # Tokenizer
    tokenizer_path = args.tokenizer_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer"
    )
    tokenizer = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))

    # Model
    config = DeeplmConfig()
    model = DeeplmModel(config)
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != config.vocab_size:
        model.resize_token_embeddings(vocab_size)
    model.to(device)
    print(f"Params: {model.num_parameters():,}")

    # Compile (CUDA only)
    if device.type == "cuda" and args.compile:
        print("Compiling model...")
        model.forward = torch.compile(model.forward, mode="default", dynamic=True)

    # Dataset
    dataset = MappedKBBIDataset(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        cache_dir=args.cache_dir,
        num_rows=args.num_rows,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # Optimizer
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.1, nesterov=True)
    total_steps = args.max_steps if args.max_steps > 0 else len(dataloader) * args.epochs
    warmup = int(total_steps * 0.03)

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(1e-5 / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_fn)

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    optimizer.zero_grad()
    global_step = 0
    start_time = time.time()
    batch_accum_loss = 0.0

    effective_bsz = args.batch_size * args.grad_accum
    print(f"\n{'='*50}")
    print(f"Training: {total_steps} steps × {effective_bsz} effective batch")
    print(f"  LR: {args.lr} → warmup={warmup} → cosine → {args.lr * lr_fn(total_steps):.2e}")
    print(f"{'='*50}")

    for epoch in range(args.epochs):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_mtp_loss=True,
            )
            loss = out["loss"] / args.grad_accum
            loss.backward()
            batch_accum_loss += out["loss"].item()

            if (global_step + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    elapsed = time.time() - start_time
                    tokens_sec = (
                        batch["input_ids"].numel() * args.grad_accum * global_step / max(elapsed, 1)
                    )
                    print(
                        f"  S{global_step:>6,} | "
                        f"L {batch_accum_loss / args.log_every:.4f} | "
                        f"LR {optimizer.param_groups[0]['lr']:.2e} | "
                        f"G {grad_norm:.2f} | "
                        f"{tokens_sec:,.0f} tok/s"
                    )
                    batch_accum_loss = 0.0

                if global_step % args.save_every == 0:
                    ckpt = os.path.join(args.output_dir, f"step-{global_step}.pt")
                    torch.save(model.state_dict(), ckpt)
                    print(f"  ✓ Saved {ckpt}")

                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    total_time = time.time() - start_time
    final = os.path.join(args.output_dir, "final.pt")
    torch.save(model.state_dict(), final)
    print(f"\nDone! {global_step} steps in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Model: {final}")

    # Upload to HF
    if args.push_to_hub:
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                print("Skip HF upload: HF_TOKEN not set")
            else:
                from huggingface_hub import HfApi, login
                login(hf_token)
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=final,
                    path_in_repo=f"checkpoints/step-{global_step}.pt",
                    repo_id=args.hub_repo_id,
                )
                print(f"Uploaded to {args.hub_repo_id}")
        except Exception as e:
            print(f"HF upload failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rows", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--output-dir", default="/output")
    parser.add_argument("--cache-dir", default="/cache")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", default="samcheng0/deeplm-108m")
    args = parser.parse_args()

    if args.num_rows == 0:
        args.num_rows = None

    train(args)
