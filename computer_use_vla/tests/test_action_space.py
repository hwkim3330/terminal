"""Tests for Computer Action Space."""

import pytest
import torch

from src.action_space.computer_action_space import (
    ComputerAction,
    ContinuousComputerActionSpace,
    DiscreteComputerActionSpace,
    ActionType,
    MouseAction,
    KeyboardAction,
    ScrollAction,
    ClickType,
)


class TestMouseAction:
    """Tests for MouseAction."""

    def test_create_mouse_action(self):
        """Test creating a mouse action."""
        action = MouseAction(x=0.5, y=0.5, click_type=ClickType.LEFT)
        assert action.x == 0.5
        assert action.y == 0.5
        assert action.click_type == ClickType.LEFT

    def test_mouse_action_to_tensor(self):
        """Test converting mouse action to tensor."""
        action = MouseAction(x=0.3, y=0.7, click_type=ClickType.RIGHT)
        tensor = action.to_tensor()
        assert tensor.shape == (5,)
        assert tensor[0].item() == pytest.approx(0.3)
        assert tensor[1].item() == pytest.approx(0.7)


class TestComputerAction:
    """Tests for ComputerAction."""

    def test_create_click_action(self):
        """Test creating a click action."""
        action = ComputerAction(
            action_type=ActionType.MOUSE_CLICK,
            mouse=MouseAction(x=0.5, y=0.5, click_type=ClickType.LEFT),
            confidence=0.95,
        )
        assert action.action_type == ActionType.MOUSE_CLICK
        assert action.mouse is not None
        assert action.confidence == 0.95

    def test_create_keyboard_action(self):
        """Test creating a keyboard action."""
        action = ComputerAction(
            action_type=ActionType.KEYBOARD_TYPE,
            keyboard=KeyboardAction(text="hello world"),
        )
        assert action.action_type == ActionType.KEYBOARD_TYPE
        assert action.keyboard.text == "hello world"

    def test_to_dict(self):
        """Test converting action to dictionary."""
        action = ComputerAction(
            action_type=ActionType.MOUSE_CLICK,
            mouse=MouseAction(x=0.5, y=0.5, click_type=ClickType.LEFT),
        )
        d = action.to_dict()
        assert d["action_type"] == "MOUSE_CLICK"
        assert "mouse" in d
        assert d["mouse"]["x"] == 0.5


class TestContinuousActionSpace:
    """Tests for ContinuousComputerActionSpace."""

    def test_create_action_space(self):
        """Test creating action space."""
        space = ContinuousComputerActionSpace(
            screen_width=1920,
            screen_height=1080,
        )
        assert space.screen_width == 1920
        assert space.screen_height == 1080

    def test_encode_decode(self):
        """Test encoding and decoding actions."""
        space = ContinuousComputerActionSpace()

        action = ComputerAction(
            action_type=ActionType.MOUSE_CLICK,
            mouse=MouseAction(x=0.5, y=0.5, click_type=ClickType.LEFT),
            confidence=1.0,
        )

        encoded = space.encode_action(action)
        decoded = space.decode_action(encoded)

        assert decoded.action_type == ActionType.MOUSE_CLICK
        assert decoded.mouse.x == pytest.approx(0.5)
        assert decoded.mouse.y == pytest.approx(0.5)

    def test_normalize_coordinates(self):
        """Test coordinate normalization."""
        space = ContinuousComputerActionSpace(
            screen_width=1920,
            screen_height=1080,
        )

        x, y = space.normalize_coordinates(960, 540)
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.5)

    def test_denormalize_coordinates(self):
        """Test coordinate denormalization."""
        space = ContinuousComputerActionSpace(
            screen_width=1920,
            screen_height=1080,
        )

        x, y = space.denormalize_coordinates(0.5, 0.5)
        assert x == 960
        assert y == 540

    def test_generate_trajectory(self):
        """Test mouse trajectory generation."""
        space = ContinuousComputerActionSpace(trajectory_length=16)

        trajectory = space.generate_mouse_trajectory(
            start_x=0.1,
            start_y=0.1,
            end_x=0.9,
            end_y=0.9,
        )

        assert trajectory.shape == (16, 2)
        # Check start and end points are close
        assert trajectory[0, 0].item() == pytest.approx(0.1, abs=0.1)
        assert trajectory[-1, 0].item() == pytest.approx(0.9, abs=0.1)


class TestDiscreteActionSpace:
    """Tests for DiscreteComputerActionSpace."""

    def test_grid_conversion(self):
        """Test grid to position conversion."""
        space = DiscreteComputerActionSpace(grid_size=10)

        # Test center position
        x, y = space.grid_to_position(55)  # Row 5, Col 5
        assert x == pytest.approx(0.55)
        assert y == pytest.approx(0.55)

        # Test reverse conversion
        idx = space.position_to_grid(0.55, 0.55)
        assert idx == 55

    def test_action_space_dims(self):
        """Test action space dimensions."""
        space = DiscreteComputerActionSpace()
        dims = space.get_action_space_dims()
        assert len(dims) == 1
        assert dims[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
