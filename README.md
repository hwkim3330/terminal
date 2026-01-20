# LFM AGI System

Computer control through human demonstration learning.

## Projects

### minVLA (Recommended)
**Karpathy-style minimal VLA - Simple, readable, hackable**

```bash
cd minvla

# Test model
python minvla.py

# Train on synthetic data
python train.py --synthetic --max-iters 1000

# Record demonstrations
python data.py --record --output demos/

# Train on real data
python train.py --data demos/

# Run demo
python demo.py --model out/best.pt
```

**Architecture (~25M params, ~1000 lines total)**:
```
Screenshot + Instruction → Transformer → (x, y, action_type)
```

See [minvla/README.md](minvla/README.md) for details.

---

### TerminaI
LFM 2.5 Instruct based terminal agent with OODA Loop reasoning.

```bash
cd terminai
python cli.py "analyze system logs"
```

### Computer Use VLA (Legacy)
LFM 2.5 VL based GUI automation agent.

---

## Quick Comparison

| Project | Style | Params | Purpose |
|---------|-------|--------|---------|
| **minVLA** | Train from scratch | ~25M | Learning & Research |
| TerminaI | API wrapper | - | Terminal automation |
| Computer Use VLA | API wrapper | - | GUI automation |

## Philosophy

> "Don't be a hero" - Andrej Karpathy

1. **Start simple** - Get something working first
2. **Data matters most** - Collect real human demonstrations
3. **Train from scratch** - Understand every component
4. **Watch the loss** - If it's not going down, something's wrong

## Installation

```bash
git clone https://github.com/hwkim3330/terminal.git
cd terminal

# Core dependencies
pip install torch numpy pillow

# For data collection & demo
pip install mss pyautogui pynput

# For API-based agents (optional)
pip install transformers aiohttp
```

## Project Structure

```
lfm_agi/
├── minvla/                    # Karpathy-style VLA (recommended)
│   ├── minvla.py              # Model (~300 lines)
│   ├── data.py                # Data collection
│   ├── train.py               # Training
│   └── demo.py                # Inference
│
├── terminai/                  # Terminal agent
│   └── src/
│       ├── core/              # OODA Loop
│       ├── pty/               # Native PTY
│       └── llm/               # LFM provider
│
└── computer_use_vla/          # GUI agent (legacy)
    └── src/
        └── models/            # VLA models
```

## Training Tips

1. **Debug with synthetic data first**
   ```bash
   python minvla/train.py --synthetic --max-iters 100
   ```

2. **Collect diverse demonstrations**
   - Different apps (browser, terminal, file manager)
   - Different tasks (click, type, scroll, drag)
   - Different screen sizes

3. **Monitor training**
   - Loss should decrease smoothly
   - Visualize predictions on test images

4. **Scale up gradually**
   - More data > more parameters
   - Pretrained vision helps (but start without)

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Karpathy's minimal GPT
- [RT-2](https://arxiv.org/abs/2307.15818) - Robotics Transformer 2
- [Alpamayo](https://developer.nvidia.com/alpamayo) - NVIDIA VLA

## License

MIT
