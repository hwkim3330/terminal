# Computer Use VLA

**Vision-Language-Action Model for Computer Control**

Inspired by NVIDIA Alpamayo architecture and Liquid AI LFM 2.5 VL.

## Overview

Computer Use VLA is an autonomous agent that uses Vision-Language-Action models to understand computer screens and execute actions to complete tasks. Similar to how NVIDIA Alpamayo controls autonomous vehicles, this system controls computers.

```
┌─────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Screen Capture  →  VL Model  →  Reasoning  →  Actions     │
│        │                │             │            │         │
│        ▼                ▼             ▼            ▼         │
│   [Screenshot]    [LFM 2.5 VL]    [Chain-of-    [Mouse/     │
│                   [Qwen-VL]        Thought]      Keyboard]   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Vision-Language Understanding**: Uses LFM 2.5 VL or Qwen-VL for screen understanding
- **Chain-of-Thought Reasoning**: Step-by-step reasoning inspired by Alpamayo's Chain-of-Causation
- **Flow Matching Trajectories**: Smooth mouse movements using diffusion-based trajectory generation
- **Multiple Backends**: PyAutoGUI, Xdotool, or dummy for testing
- **Safety Features**: Blocks dangerous actions, supports dry-run mode

## Installation

```bash
# Clone the repository
cd /mnt/data/lfm_agi/computer_use_vla

# Install with pip
pip install -e .

# With controller support
pip install -e ".[controller]"

# With LFM model support
pip install -e ".[lfm]"

# All dependencies
pip install -e ".[all]"
```

## Quick Start

### Basic Usage

```python
from computer_use_vla import create_agent

# Create agent
agent = create_agent(
    model="LiquidAI/LFM2.5-VL-1.6B",
    backend="pyautogui",
)

# Run a task
result = agent.run("Open Chrome and search for 'hello world'")

print(f"Success: {result['success']}")
print(f"Steps: {result['steps']}")
```

### Interactive Mode

```python
from computer_use_vla import create_agent

# Create interactive agent (confirms each action)
agent = create_agent(interactive=True)

# Run with confirmation
result = agent.run("Delete the file on desktop")
```

### Dry Run (Testing)

```python
from computer_use_vla import create_agent

# Create agent in dry run mode
agent = create_agent(dry_run=True, backend="dummy")

# Test without actual execution
result = agent.run("Click the button")
```

## Command Line

```bash
# Run with a task
python examples/run_agent.py --task "Open the browser"

# Interactive mode
python examples/run_agent.py --interactive --task "Navigate to Documents"

# Dry run
python examples/run_agent.py --task "Click button" --dry-run

# Demo mode
python examples/run_agent.py --demo
```

## Architecture

### Inspired by NVIDIA Alpamayo

| Alpamayo (Autonomous Driving) | Computer Use VLA |
|------------------------------|------------------|
| Camera images | Screen captures |
| Road detection | UI element detection |
| Trajectory planning | Mouse trajectory planning |
| Steering/Throttle | Mouse/Keyboard actions |
| Chain-of-Causation | Chain-of-Thought |
| Flow Matching decoder | Flow Matching for mouse paths |

### Components

1. **Action Space** (`src/action_space/`)
   - Mouse actions (move, click, drag)
   - Keyboard actions (type, hotkey)
   - Scroll actions
   - Discrete and continuous variants

2. **VLA Model** (`src/models/`)
   - Vision encoder for screen understanding
   - Language model for reasoning
   - Action decoder for computer control

3. **Controller** (`src/controller/`)
   - PyAutoGUI backend (cross-platform)
   - Xdotool backend (Linux)
   - Dummy backend (testing)

4. **Flow Matching** (`src/diffusion/`)
   - Smooth mouse trajectory generation
   - Action sequence planning

## Configuration

See `configs/default_config.yaml` for all options:

```yaml
model:
  name: "LiquidAI/LFM2.5-VL-1.6B"
  dtype: "float16"

reasoning:
  enabled: true
  max_tokens: 512

controller:
  backend: "pyautogui"
  safe_mode: true
  dry_run: false

agent:
  max_steps: 50
  step_delay: 0.5
```

## Safety

- **Safe Mode**: Blocks dangerous key combinations (Ctrl+Alt+Delete, etc.)
- **Dry Run**: Test without execution
- **Interactive Mode**: Human confirmation for each action
- **Max Steps**: Limits agent iterations

## Supported Models

| Model | Size | Description |
|-------|------|-------------|
| LiquidAI/LFM2.5-VL-1.6B | 1.6B | Lightweight, edge-friendly |
| Qwen/Qwen2-VL-7B-Instruct | 7B | More powerful |

## References

- [NVIDIA Alpamayo](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/) - VLA for autonomous driving
- [Liquid AI LFM 2.5](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai) - On-device AI models
- [Flow Matching](https://arxiv.org/pdf/2210.02747) - Generative modeling

## License

Apache-2.0
