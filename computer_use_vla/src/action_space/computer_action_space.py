# SPDX-License-Identifier: Apache-2.0
# Computer Use VLA - Action Space Definition
# Inspired by NVIDIA Alpamayo architecture

"""Computer Action Space for Vision-Language-Action models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Tuple, List

import torch
from torch import nn
import numpy as np


class ActionType(Enum):
    """Types of computer actions."""
    MOUSE_MOVE = auto()
    MOUSE_CLICK = auto()
    MOUSE_DRAG = auto()
    MOUSE_SCROLL = auto()
    KEYBOARD_TYPE = auto()
    KEYBOARD_KEY = auto()
    KEYBOARD_HOTKEY = auto()
    WAIT = auto()
    SCREENSHOT = auto()
    DONE = auto()


class ClickType(Enum):
    """Mouse click types."""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"
    DOUBLE = "double"


@dataclass
class MouseAction:
    """Mouse action representation."""
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    click_type: Optional[ClickType] = None
    drag_end_x: Optional[float] = None
    drag_end_y: Optional[float] = None

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor [x, y, click_type_id, drag_end_x, drag_end_y]."""
        click_id = list(ClickType).index(self.click_type) if self.click_type else -1
        return torch.tensor([
            self.x, self.y, click_id,
            self.drag_end_x or -1,
            self.drag_end_y or -1
        ], dtype=torch.float32)


@dataclass
class KeyboardAction:
    """Keyboard action representation."""
    text: Optional[str] = None  # For typing text
    key: Optional[str] = None   # For single key press
    modifiers: Optional[List[str]] = None  # ctrl, alt, shift, etc.

    def to_tensor(self, tokenizer=None, max_length: int = 64) -> torch.Tensor:
        """Convert to tensor representation."""
        if self.text and tokenizer:
            tokens = tokenizer.encode(self.text, max_length=max_length)
            return torch.tensor(tokens, dtype=torch.long)
        return torch.zeros(max_length, dtype=torch.long)


@dataclass
class ScrollAction:
    """Scroll action representation."""
    direction: str  # "up", "down", "left", "right"
    amount: float   # Scroll amount (normalized)

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor [direction_id, amount]."""
        directions = ["up", "down", "left", "right"]
        dir_id = directions.index(self.direction) if self.direction in directions else 0
        return torch.tensor([dir_id, self.amount], dtype=torch.float32)


@dataclass
class ComputerAction:
    """Unified computer action representation."""
    action_type: ActionType
    mouse: Optional[MouseAction] = None
    keyboard: Optional[KeyboardAction] = None
    scroll: Optional[ScrollAction] = None
    wait_duration: Optional[float] = None  # seconds
    reasoning: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary for execution."""
        result = {
            "action_type": self.action_type.name,
            "confidence": self.confidence,
        }
        if self.mouse:
            result["mouse"] = {
                "x": self.mouse.x,
                "y": self.mouse.y,
                "click_type": self.mouse.click_type.value if self.mouse.click_type else None,
                "drag_end": (self.mouse.drag_end_x, self.mouse.drag_end_y) if self.mouse.drag_end_x else None
            }
        if self.keyboard:
            result["keyboard"] = {
                "text": self.keyboard.text,
                "key": self.keyboard.key,
                "modifiers": self.keyboard.modifiers
            }
        if self.scroll:
            result["scroll"] = {
                "direction": self.scroll.direction,
                "amount": self.scroll.amount
            }
        if self.wait_duration:
            result["wait_duration"] = self.wait_duration
        if self.reasoning:
            result["reasoning"] = self.reasoning
        return result


class ComputerActionSpace(ABC, nn.Module):
    """
    Action space base class for computer use.
    Similar to Alpamayo's ActionSpace but for computer control.
    """

    # Screen dimensions for coordinate normalization
    DEFAULT_SCREEN_WIDTH = 1920
    DEFAULT_SCREEN_HEIGHT = 1080

    # Action space dimensions
    MOUSE_DIMS = 5      # x, y, click_type, drag_x, drag_y
    KEYBOARD_DIMS = 64  # Token sequence
    SCROLL_DIMS = 2     # direction, amount
    META_DIMS = 3       # action_type, confidence, wait_duration

    def __init__(
        self,
        screen_width: int = DEFAULT_SCREEN_WIDTH,
        screen_height: int = DEFAULT_SCREEN_HEIGHT,
        max_text_length: int = 64,
    ):
        super().__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_text_length = max_text_length

    @abstractmethod
    def encode_action(self, action: ComputerAction) -> torch.Tensor:
        """Encode computer action to tensor."""
        pass

    @abstractmethod
    def decode_action(self, tensor: torch.Tensor) -> ComputerAction:
        """Decode tensor to computer action."""
        pass

    def get_action_space_dims(self) -> Tuple[int, ...]:
        """Get the dimensions of the action space."""
        return (self.META_DIMS + self.MOUSE_DIMS + self.SCROLL_DIMS,)

    def normalize_coordinates(self, x: int, y: int) -> Tuple[float, float]:
        """Normalize screen coordinates to 0-1 range."""
        return x / self.screen_width, y / self.screen_height

    def denormalize_coordinates(self, x: float, y: float) -> Tuple[int, int]:
        """Convert normalized coordinates back to screen pixels."""
        return int(x * self.screen_width), int(y * self.screen_height)

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        """Check if the action coordinates are within screen bounds."""
        # Extract normalized coordinates
        x, y = action[..., 0], action[..., 1]
        in_bounds = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        return in_bounds


class DiscreteComputerActionSpace(ComputerActionSpace):
    """
    Discrete action space for computer use.
    Actions are discretized into grid cells for mouse position.
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        grid_size: int = 100,  # 100x100 grid
        max_text_length: int = 64,
        vocab_size: int = 256,  # For keyboard tokenization
    ):
        super().__init__(screen_width, screen_height, max_text_length)
        self.grid_size = grid_size
        self.vocab_size = vocab_size

        # Embedding layers
        self.position_embedding = nn.Embedding(grid_size * grid_size, 128)
        self.action_type_embedding = nn.Embedding(len(ActionType), 64)
        self.click_type_embedding = nn.Embedding(len(ClickType) + 1, 32)  # +1 for no click

    def encode_action(self, action: ComputerAction) -> torch.Tensor:
        """Encode computer action to tensor."""
        # Initialize tensor
        tensor = torch.zeros(self.META_DIMS + self.MOUSE_DIMS + self.SCROLL_DIMS)

        # Meta info
        tensor[0] = action.action_type.value
        tensor[1] = action.confidence
        tensor[2] = action.wait_duration or 0.0

        # Mouse info
        if action.mouse:
            tensor[3] = action.mouse.x
            tensor[4] = action.mouse.y
            tensor[5] = list(ClickType).index(action.mouse.click_type) + 1 if action.mouse.click_type else 0
            tensor[6] = action.mouse.drag_end_x or 0.0
            tensor[7] = action.mouse.drag_end_y or 0.0

        # Scroll info
        if action.scroll:
            directions = ["up", "down", "left", "right"]
            tensor[8] = directions.index(action.scroll.direction) if action.scroll.direction in directions else 0
            tensor[9] = action.scroll.amount

        return tensor

    def decode_action(self, tensor: torch.Tensor) -> ComputerAction:
        """Decode tensor to computer action."""
        action_type_id = int(tensor[0].item())
        confidence = tensor[1].item()
        wait_duration = tensor[2].item() if tensor[2].item() > 0 else None

        # Get action type
        action_type = ActionType(action_type_id) if 0 < action_type_id <= len(ActionType) else ActionType.MOUSE_MOVE

        # Decode mouse action
        mouse = None
        if action_type in [ActionType.MOUSE_MOVE, ActionType.MOUSE_CLICK, ActionType.MOUSE_DRAG]:
            click_type_id = int(tensor[5].item())
            mouse = MouseAction(
                x=tensor[3].item(),
                y=tensor[4].item(),
                click_type=list(ClickType)[click_type_id - 1] if click_type_id > 0 else None,
                drag_end_x=tensor[6].item() if tensor[6].item() > 0 else None,
                drag_end_y=tensor[7].item() if tensor[7].item() > 0 else None,
            )

        # Decode scroll action
        scroll = None
        if action_type == ActionType.MOUSE_SCROLL:
            directions = ["up", "down", "left", "right"]
            dir_id = int(tensor[8].item())
            scroll = ScrollAction(
                direction=directions[dir_id] if 0 <= dir_id < 4 else "down",
                amount=tensor[9].item()
            )

        return ComputerAction(
            action_type=action_type,
            mouse=mouse,
            scroll=scroll,
            wait_duration=wait_duration,
            confidence=confidence,
        )

    def grid_to_position(self, grid_idx: int) -> Tuple[float, float]:
        """Convert grid index to normalized position."""
        row = grid_idx // self.grid_size
        col = grid_idx % self.grid_size
        x = (col + 0.5) / self.grid_size
        y = (row + 0.5) / self.grid_size
        return x, y

    def position_to_grid(self, x: float, y: float) -> int:
        """Convert normalized position to grid index."""
        col = min(int(x * self.grid_size), self.grid_size - 1)
        row = min(int(y * self.grid_size), self.grid_size - 1)
        return row * self.grid_size + col


class ContinuousComputerActionSpace(ComputerActionSpace):
    """
    Continuous action space for computer use.
    Uses continuous coordinates for precise positioning.
    Similar to Alpamayo's trajectory-based action space.
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        max_text_length: int = 64,
        trajectory_length: int = 16,  # Number of waypoints for mouse trajectory
    ):
        super().__init__(screen_width, screen_height, max_text_length)
        self.trajectory_length = trajectory_length

    def encode_action(self, action: ComputerAction) -> torch.Tensor:
        """Encode computer action to continuous tensor."""
        tensor = torch.zeros(self.META_DIMS + self.MOUSE_DIMS + self.SCROLL_DIMS)

        tensor[0] = action.action_type.value
        tensor[1] = action.confidence
        tensor[2] = action.wait_duration or 0.0

        if action.mouse:
            tensor[3] = action.mouse.x
            tensor[4] = action.mouse.y
            tensor[5] = list(ClickType).index(action.mouse.click_type) + 1 if action.mouse.click_type else 0
            tensor[6] = action.mouse.drag_end_x or action.mouse.x
            tensor[7] = action.mouse.drag_end_y or action.mouse.y

        if action.scroll:
            directions = ["up", "down", "left", "right"]
            tensor[8] = directions.index(action.scroll.direction) if action.scroll.direction in directions else 0
            tensor[9] = action.scroll.amount

        return tensor

    def decode_action(self, tensor: torch.Tensor) -> ComputerAction:
        """Decode continuous tensor to computer action."""
        return DiscreteComputerActionSpace.decode_action(self, tensor)

    def generate_mouse_trajectory(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> torch.Tensor:
        """
        Generate smooth mouse trajectory from start to end.
        Uses cubic bezier interpolation for natural movement.

        Returns:
            trajectory: [trajectory_length, 2] tensor of (x, y) positions
        """
        t = torch.linspace(0, 1, self.trajectory_length)

        # Add slight curve for natural movement
        mid_x = (start_x + end_x) / 2 + (np.random.random() - 0.5) * 0.1
        mid_y = (start_y + end_y) / 2 + (np.random.random() - 0.5) * 0.1

        # Quadratic bezier curve
        x = (1 - t) ** 2 * start_x + 2 * (1 - t) * t * mid_x + t ** 2 * end_x
        y = (1 - t) ** 2 * start_y + 2 * (1 - t) * t * mid_y + t ** 2 * end_y

        trajectory = torch.stack([x, y], dim=-1)
        return trajectory

    def trajectory_to_action(
        self,
        trajectory: torch.Tensor,
        click_at_end: bool = True,
        click_type: ClickType = ClickType.LEFT,
    ) -> ComputerAction:
        """Convert trajectory to computer action."""
        end_pos = trajectory[-1]

        return ComputerAction(
            action_type=ActionType.MOUSE_CLICK if click_at_end else ActionType.MOUSE_MOVE,
            mouse=MouseAction(
                x=end_pos[0].item(),
                y=end_pos[1].item(),
                click_type=click_type if click_at_end else None,
            ),
            confidence=1.0,
        )
