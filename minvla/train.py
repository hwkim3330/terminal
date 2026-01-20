"""
Training Script for MiniVLA
============================

Simple, readable training loop. No fancy frameworks.
Just PyTorch, a model, data, and gradient descent.

Usage:
    # Train on synthetic data (for testing)
    python train.py --synthetic

    # Train on real demos
    python train.py --data demos/

    # Resume training
    python train.py --data demos/ --resume checkpoints/latest.pt

Key insights from Karpathy:
1. Start simple, verify everything works
2. Use synthetic data first to debug
3. Watch the loss curves
4. Don't tune hyperparameters until your data is solid

Author: Karpathy-style refactor
"""

import os
import time
import math
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from minvla import MiniVLA, VLAConfig, compute_loss
from data import SyntheticDataset, VLADataset


# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data', type=str, default=None, help='Path to training data')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--num-synthetic', type=int, default=50000)

    # Model
    parser.add_argument('--n-embd', type=int, default=512)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-layer', type=int, default=6)

    # Training
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-iters', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-iters', type=int, default=100)
    parser.add_argument('--lr-decay-iters', type=int, default=10000)
    parser.add_argument('--min-lr', type=float, default=3e-5)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--dtype', type=str, default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')

    # Logging
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--eval-iters', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=1000)

    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Learning Rate Schedule
# -----------------------------------------------------------------------------

def get_lr(it, args):
    """
    Learning rate schedule with warmup and cosine decay.

    1. Linear warmup for warmup_iters
    2. Cosine decay to min_lr
    """
    # Warmup
    if it < args.warmup_iters:
        return args.lr * it / args.warmup_iters

    # After decay
    if it > args.lr_decay_iters:
        return args.min_lr

    # Cosine decay
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

def train(args):
    print("="*60)
    print("MiniVLA Training")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Data: {'synthetic' if args.synthetic else args.data}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max iters: {args.max_iters}")
    print(f"Learning rate: {args.lr}")
    print("="*60)

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    # Precision
    dtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[args.dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=dtype)
    scaler = torch.amp.GradScaler(device, enabled=(args.dtype == 'float16'))

    # Data
    if args.synthetic:
        train_dataset = SyntheticDataset(num_samples=args.num_synthetic)
        val_dataset = SyntheticDataset(num_samples=1000)
    else:
        if not args.data:
            print("Error: --data required when not using --synthetic")
            return

        # Simple tokenizer (just use character-level for now)
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) % 50257 for c in text[:128]]
            def decode(self, tokens):
                return ''.join(chr(t % 128) for t in tokens)

        tokenizer = SimpleTokenizer()
        train_dataset = VLADataset(args.data, tokenizer, augment=True)
        val_dataset = VLADataset(args.data, tokenizer, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    train_iter = iter(train_loader)

    # Model
    config = VLAConfig(
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer
    )
    model = MiniVLA(config).to(device)

    # Optimizer
    # Separate weight decay for different parameter groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'ln' in name or 'emb' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))

    # Compile
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)

    # Resume
    start_iter = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iter']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Training loop
    print("\nStarting training...")
    print("-"*60)

    model.train()
    t0 = time.time()
    running_loss = 0.0

    for it in range(start_iter, args.max_iters):
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        img = batch['image'].to(device)
        tokens = batch['tokens'].to(device)
        target_coords = batch['coords'].to(device)
        target_type = batch['action_type'].to(device)

        # Learning rate schedule
        lr = get_lr(it, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        with ctx:
            pred = model(img, tokens)
            target = {
                'coords': target_coords,
                'action_type': target_type
            }
            loss, metrics = compute_loss(pred, target)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()

        # Logging
        if it % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            avg_loss = running_loss / args.log_interval
            running_loss = 0.0

            print(f"iter {it:5d} | loss {avg_loss:.4f} | "
                  f"coord {metrics['coord_loss']:.4f} | type {metrics['type_loss']:.4f} | "
                  f"lr {lr:.2e} | {dt*1000/args.log_interval:.1f}ms/iter")

        # Evaluation
        if it > 0 and it % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, ctx, args.eval_iters)
            print(f">>> val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, it, best_val_loss, args.out_dir, 'best.pt')
                print(f">>> new best model saved!")

            model.train()

        # Save checkpoint
        if it > 0 and it % args.save_interval == 0:
            save_checkpoint(model, optimizer, it, best_val_loss, args.out_dir, 'latest.pt')
            save_checkpoint(model, optimizer, it, best_val_loss, args.out_dir, f'iter_{it}.pt')

    # Final save
    save_checkpoint(model, optimizer, args.max_iters, best_val_loss, args.out_dir, 'final.pt')
    print("\nTraining complete!")


@torch.no_grad()
def evaluate(model, val_loader, device, ctx, max_iters):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if i >= max_iters:
            break

        img = batch['image'].to(device)
        tokens = batch['tokens'].to(device)
        target_coords = batch['coords'].to(device)
        target_type = batch['action_type'].to(device)

        with ctx:
            pred = model(img, tokens)
            target = {
                'coords': target_coords,
                'action_type': target_type
            }
            loss, _ = compute_loss(pred, target)

        total_loss += loss.item()
        count += 1

    return total_loss / count if count > 0 else float('inf')


def save_checkpoint(model, optimizer, it, best_val_loss, out_dir, filename):
    """Save training checkpoint."""
    checkpoint = {
        'model': model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter': it,
        'best_val_loss': best_val_loss
    }
    path = Path(out_dir) / filename
    torch.save(checkpoint, path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    args = get_args()
    train(args)
