# minVLA

Minimal Vision-Language-Action model for computer control.

Inspired by [Andrej Karpathy's](https://karpathy.ai/) philosophy: simple, readable, hackable.

## What is this?

A ~500 line implementation of a VLA (Vision-Language-Action) model that can learn to control a computer from human demonstrations.

```
Screenshot + "click the search button" → (x=450, y=230, action=click)
```

## Files

```
minvla/
├── minvla.py   # Model (~300 lines)
├── data.py     # Data collection & loading (~200 lines)
├── train.py    # Training script (~200 lines)
└── demo.py     # Inference & demo (~200 lines)
```

## Quickstart

```bash
# 1. Install
pip install torch numpy pillow

# Optional (for data collection & demo)
pip install mss pyautogui pynput

# 2. Test model
python minvla.py

# 3. Train on synthetic data
python train.py --synthetic --max-iters 1000

# 4. Record real demonstrations
python data.py --record --output demos/

# 5. Train on real data
python train.py --data demos/ --max-iters 10000

# 6. Run demo
python demo.py --model out/best.pt
```

## Model Architecture

```
┌─────────────┐   ┌─────────────┐
│ Screenshot  │   │ Instruction │
│  224x224    │   │   "click.." │
└──────┬──────┘   └──────┬──────┘
       │                 │
       ▼                 ▼
  ┌─────────┐      ┌─────────┐
  │  Patch  │      │  Token  │
  │  Embed  │      │  Embed  │
  └────┬────┘      └────┬────┘
       │                 │
       └────────┬────────┘
                │
                ▼
         ┌───────────┐
         │Transformer│
         │ 6 layers  │
         └─────┬─────┘
               │
               ▼
         ┌───────────┐
         │Action Head│
         └─────┬─────┘
               │
               ▼
    ┌──────────────────────┐
    │ x, y, type, conf     │
    │ (click at 450, 230)  │
    └──────────────────────┘
```

**Parameters**: ~25M (configurable)

## Key Ideas

### 1. Data is Everything

```python
# Record human demonstrations
recorder = DemoRecorder("demos/")
recorder.start()
recorder.set_instruction("click the search button")
# ... human performs action ...
recorder.stop()
```

No amount of architecture can substitute for good data.

### 2. Simple Architecture

- Patch embedding (no pretrained ViT)
- Standard transformer (no fancy attention)
- Direct coordinate regression
- Single forward pass

### 3. Train from Scratch

```python
# Training loop - that's it
for batch in dataloader:
    pred = model(batch['image'], batch['tokens'])
    loss = compute_loss(pred, batch)
    loss.backward()
    optimizer.step()
```

## Action Space

| Type | Description |
|------|-------------|
| click | Left click at (x, y) |
| double_click | Double click |
| right_click | Right click |
| drag | Drag from (x,y) to (end_x, end_y) |
| scroll | Scroll at position |
| type | Type text |
| hotkey | Press key combination |

## Training Tips

1. **Start with synthetic data** - Debug your setup
2. **Collect diverse demos** - Different apps, different tasks
3. **Watch the loss** - Should decrease smoothly
4. **Validate visually** - Look at predictions on test images

## Differences from Production VLAs

| This Project | Production |
|--------------|------------|
| Train from scratch | Pretrained vision encoder |
| ~25M params | 3B+ params |
| Simple tokenizer | SentencePiece/BPE |
| Direct regression | Discrete tokens |
| No history | Multi-turn context |

## References

- [RT-2](https://arxiv.org/abs/2307.15818) - Robotics Transformer 2
- [Alpamayo](https://developer.nvidia.com/blog/advancing-autonomous-vehicle-development-with-end-to-end-learning/) - NVIDIA's VLA
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Karpathy's minimal GPT

## License

MIT
