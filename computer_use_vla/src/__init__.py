# Computer Use VLA
# Vision-Language-Action model for computer control
# Inspired by NVIDIA Alpamayo architecture

from .action_space.computer_action_space import (
    ComputerAction,
    ComputerActionSpace,
    ContinuousComputerActionSpace,
    DiscreteComputerActionSpace,
    ActionType,
    MouseAction,
    KeyboardAction,
    ScrollAction,
    ClickType,
)

from .models.computer_use_vla import (
    ComputerUseVLA,
    ComputerUseVLAConfig,
    ComputerUseOutput,
    ChainOfThoughtReasoning,
    ActionDecoder,
)

from .controller.computer_controller import (
    ComputerController,
    ControllerConfig,
    ActionResult,
)

from .diffusion.action_flow_matching import (
    ActionFlowMatching,
    MouseTrajectoryGenerator,
    ActionSequenceGenerator,
)

__version__ = "0.1.0"
__all__ = [
    # Action Space
    "ComputerAction",
    "ComputerActionSpace",
    "ContinuousComputerActionSpace",
    "DiscreteComputerActionSpace",
    "ActionType",
    "MouseAction",
    "KeyboardAction",
    "ScrollAction",
    "ClickType",
    # Model
    "ComputerUseVLA",
    "ComputerUseVLAConfig",
    "ComputerUseOutput",
    "ChainOfThoughtReasoning",
    "ActionDecoder",
    # Controller
    "ComputerController",
    "ControllerConfig",
    "ActionResult",
    # Diffusion
    "ActionFlowMatching",
    "MouseTrajectoryGenerator",
    "ActionSequenceGenerator",
]
