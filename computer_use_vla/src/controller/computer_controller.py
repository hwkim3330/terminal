# SPDX-License-Identifier: Apache-2.0
# Computer Use VLA - Computer Controller
# Executes actions on the actual computer

"""
Computer Controller for executing VLA actions.

Supports multiple backends:
- PyAutoGUI (cross-platform)
- Xdotool (Linux)
- AppleScript (macOS)
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
import subprocess

from ..action_space.computer_action_space import (
    ComputerAction,
    ActionType,
    MouseAction,
    KeyboardAction,
    ScrollAction,
    ClickType,
)

logger = logging.getLogger(__name__)


@dataclass
class ControllerConfig:
    """Configuration for computer controller."""
    screen_width: int = 1920
    screen_height: int = 1080
    mouse_move_duration: float = 0.3  # seconds
    click_delay: float = 0.1  # seconds between clicks
    typing_interval: float = 0.05  # seconds between keystrokes
    safe_mode: bool = True  # Require confirmation for dangerous actions
    dry_run: bool = False  # Don't actually execute actions
    backend: str = "pyautogui"  # "pyautogui", "xdotool", "dummy"


class ActionResult:
    """Result of an action execution."""
    def __init__(
        self,
        success: bool,
        action: ComputerAction,
        message: str = "",
        screenshot_after: Optional[bytes] = None,
    ):
        self.success = success
        self.action = action
        self.message = message
        self.screenshot_after = screenshot_after
        self.timestamp = time.time()


class ControllerBackend(ABC):
    """Abstract base class for controller backends."""

    @abstractmethod
    def move_mouse(self, x: int, y: int, duration: float = 0.3) -> bool:
        pass

    @abstractmethod
    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        pass

    @abstractmethod
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> bool:
        pass

    @abstractmethod
    def scroll(self, amount: int, direction: str = "down") -> bool:
        pass

    @abstractmethod
    def type_text(self, text: str, interval: float = 0.05) -> bool:
        pass

    @abstractmethod
    def press_key(self, key: str, modifiers: Optional[List[str]] = None) -> bool:
        pass

    @abstractmethod
    def screenshot(self) -> bytes:
        pass

    @abstractmethod
    def get_screen_size(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_mouse_position(self) -> Tuple[int, int]:
        pass


class PyAutoGUIBackend(ControllerBackend):
    """PyAutoGUI backend for cross-platform support."""

    def __init__(self, config: ControllerConfig):
        self.config = config
        try:
            import pyautogui
            self.pyautogui = pyautogui
            # Safety settings
            pyautogui.FAILSAFE = True  # Move mouse to corner to abort
            pyautogui.PAUSE = 0.1
        except ImportError:
            logger.warning("PyAutoGUI not installed. Run: pip install pyautogui")
            self.pyautogui = None

    def move_mouse(self, x: int, y: int, duration: float = 0.3) -> bool:
        if self.pyautogui is None:
            return False
        try:
            self.pyautogui.moveTo(x, y, duration=duration)
            return True
        except Exception as e:
            logger.error(f"Mouse move failed: {e}")
            return False

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        if self.pyautogui is None:
            return False
        try:
            self.pyautogui.click(x, y, button=button, clicks=clicks)
            return True
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> bool:
        if self.pyautogui is None:
            return False
        try:
            self.pyautogui.moveTo(start_x, start_y)
            self.pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
            return True
        except Exception as e:
            logger.error(f"Drag failed: {e}")
            return False

    def scroll(self, amount: int, direction: str = "down") -> bool:
        if self.pyautogui is None:
            return False
        try:
            clicks = -amount if direction == "down" else amount
            self.pyautogui.scroll(clicks)
            return True
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return False

    def type_text(self, text: str, interval: float = 0.05) -> bool:
        if self.pyautogui is None:
            return False
        try:
            self.pyautogui.typewrite(text, interval=interval)
            return True
        except Exception as e:
            logger.error(f"Type failed: {e}")
            return False

    def press_key(self, key: str, modifiers: Optional[List[str]] = None) -> bool:
        if self.pyautogui is None:
            return False
        try:
            if modifiers:
                self.pyautogui.hotkey(*modifiers, key)
            else:
                self.pyautogui.press(key)
            return True
        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return False

    def screenshot(self) -> bytes:
        if self.pyautogui is None:
            return b""
        try:
            import io
            screenshot = self.pyautogui.screenshot()
            buffer = io.BytesIO()
            screenshot.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return b""

    def get_screen_size(self) -> Tuple[int, int]:
        if self.pyautogui is None:
            return (self.config.screen_width, self.config.screen_height)
        return self.pyautogui.size()

    def get_mouse_position(self) -> Tuple[int, int]:
        if self.pyautogui is None:
            return (0, 0)
        return self.pyautogui.position()


class XdotoolBackend(ControllerBackend):
    """Xdotool backend for Linux."""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self._check_xdotool()

    def _check_xdotool(self) -> bool:
        try:
            subprocess.run(["xdotool", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("xdotool not found. Install with: sudo apt install xdotool")
            return False

    def _run_xdotool(self, *args) -> bool:
        try:
            subprocess.run(["xdotool", *args], check=True)
            return True
        except Exception as e:
            logger.error(f"xdotool command failed: {e}")
            return False

    def move_mouse(self, x: int, y: int, duration: float = 0.3) -> bool:
        return self._run_xdotool("mousemove", str(x), str(y))

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        button_map = {"left": "1", "middle": "2", "right": "3"}
        btn = button_map.get(button, "1")
        self.move_mouse(x, y)
        for _ in range(clicks):
            if not self._run_xdotool("click", btn):
                return False
        return True

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> bool:
        self.move_mouse(start_x, start_y)
        self._run_xdotool("mousedown", "1")
        self.move_mouse(end_x, end_y)
        return self._run_xdotool("mouseup", "1")

    def scroll(self, amount: int, direction: str = "down") -> bool:
        button = "5" if direction == "down" else "4"
        for _ in range(abs(amount)):
            if not self._run_xdotool("click", button):
                return False
        return True

    def type_text(self, text: str, interval: float = 0.05) -> bool:
        return self._run_xdotool("type", "--delay", str(int(interval * 1000)), text)

    def press_key(self, key: str, modifiers: Optional[List[str]] = None) -> bool:
        if modifiers:
            key_combo = "+".join(modifiers + [key])
            return self._run_xdotool("key", key_combo)
        return self._run_xdotool("key", key)

    def screenshot(self) -> bytes:
        try:
            import io
            from PIL import Image
            result = subprocess.run(
                ["import", "-window", "root", "png:-"],
                capture_output=True,
                check=True
            )
            return result.stdout
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return b""

    def get_screen_size(self) -> Tuple[int, int]:
        try:
            result = subprocess.run(
                ["xdpyinfo"],
                capture_output=True,
                text=True,
                check=True
            )
            import re
            match = re.search(r'dimensions:\s+(\d+)x(\d+)', result.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
        except Exception:
            pass
        return self.config.screen_width, self.config.screen_height

    def get_mouse_position(self) -> Tuple[int, int]:
        try:
            result = subprocess.run(
                ["xdotool", "getmouselocation"],
                capture_output=True,
                text=True,
                check=True
            )
            import re
            match = re.search(r'x:(\d+)\s+y:(\d+)', result.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
        except Exception:
            pass
        return 0, 0


class DummyBackend(ControllerBackend):
    """Dummy backend for testing without actual control."""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.mouse_x = 0
        self.mouse_y = 0
        self.action_log = []

    def _log_action(self, action: str):
        self.action_log.append({"action": action, "time": time.time()})
        logger.info(f"[DUMMY] {action}")

    def move_mouse(self, x: int, y: int, duration: float = 0.3) -> bool:
        self._log_action(f"Mouse move to ({x}, {y})")
        self.mouse_x, self.mouse_y = x, y
        return True

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        self._log_action(f"Click {button} at ({x}, {y}) x{clicks}")
        return True

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> bool:
        self._log_action(f"Drag from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        return True

    def scroll(self, amount: int, direction: str = "down") -> bool:
        self._log_action(f"Scroll {direction} by {amount}")
        return True

    def type_text(self, text: str, interval: float = 0.05) -> bool:
        self._log_action(f"Type: '{text}'")
        return True

    def press_key(self, key: str, modifiers: Optional[List[str]] = None) -> bool:
        mod_str = "+".join(modifiers) + "+" if modifiers else ""
        self._log_action(f"Press key: {mod_str}{key}")
        return True

    def screenshot(self) -> bytes:
        self._log_action("Screenshot")
        # Return a 1x1 black PNG
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'

    def get_screen_size(self) -> Tuple[int, int]:
        return self.config.screen_width, self.config.screen_height

    def get_mouse_position(self) -> Tuple[int, int]:
        return self.mouse_x, self.mouse_y


class ComputerController:
    """
    Main computer controller for executing VLA actions.

    Features:
    - Multiple backend support
    - Action safety checks
    - Execution logging
    - Screenshot feedback
    """

    # Dangerous key combinations that require confirmation
    DANGEROUS_HOTKEYS = [
        ("ctrl", "alt", "delete"),
        ("alt", "f4"),
        ("ctrl", "shift", "q"),
        ("super", "l"),  # Lock screen
    ]

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()

        # Initialize backend
        self.backend = self._create_backend()

        # Get actual screen size
        self.screen_width, self.screen_height = self.backend.get_screen_size()

        # Callbacks
        self.on_before_action: Optional[Callable[[ComputerAction], bool]] = None
        self.on_after_action: Optional[Callable[[ActionResult], None]] = None

        logger.info(f"ComputerController initialized with {self.config.backend} backend")
        logger.info(f"Screen size: {self.screen_width}x{self.screen_height}")

    def _create_backend(self) -> ControllerBackend:
        """Create appropriate backend."""
        if self.config.backend == "xdotool":
            return XdotoolBackend(self.config)
        elif self.config.backend == "dummy":
            return DummyBackend(self.config)
        else:
            return PyAutoGUIBackend(self.config)

    def denormalize_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Convert normalized (0-1) coordinates to screen pixels."""
        px = int(x * self.screen_width)
        py = int(y * self.screen_height)
        # Clamp to screen bounds
        px = max(0, min(px, self.screen_width - 1))
        py = max(0, min(py, self.screen_height - 1))
        return px, py

    def is_action_safe(self, action: ComputerAction) -> Tuple[bool, str]:
        """Check if action is safe to execute."""
        if action.action_type == ActionType.KEYBOARD_KEY and action.keyboard:
            key = action.keyboard.key
            mods = action.keyboard.modifiers or []

            # Check for dangerous combinations
            for dangerous in self.DANGEROUS_HOTKEYS:
                if set(mods + [key]) == set(dangerous):
                    return False, f"Dangerous hotkey detected: {'+'.join(dangerous)}"

        return True, "Safe"

    def execute(self, action: ComputerAction) -> ActionResult:
        """Execute a computer action."""
        # Safety check
        if self.config.safe_mode:
            is_safe, message = self.is_action_safe(action)
            if not is_safe:
                return ActionResult(
                    success=False,
                    action=action,
                    message=f"Action blocked: {message}",
                )

        # Pre-action callback
        if self.on_before_action:
            if not self.on_before_action(action):
                return ActionResult(
                    success=False,
                    action=action,
                    message="Action cancelled by callback",
                )

        # Dry run mode
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would execute: {action.to_dict()}")
            return ActionResult(
                success=True,
                action=action,
                message="Dry run - action not executed",
            )

        # Execute action based on type
        success = False
        message = ""

        try:
            if action.action_type == ActionType.MOUSE_MOVE:
                success = self._execute_mouse_move(action)
            elif action.action_type == ActionType.MOUSE_CLICK:
                success = self._execute_mouse_click(action)
            elif action.action_type == ActionType.MOUSE_DRAG:
                success = self._execute_mouse_drag(action)
            elif action.action_type == ActionType.MOUSE_SCROLL:
                success = self._execute_scroll(action)
            elif action.action_type == ActionType.KEYBOARD_TYPE:
                success = self._execute_keyboard_type(action)
            elif action.action_type == ActionType.KEYBOARD_KEY:
                success = self._execute_keyboard_key(action)
            elif action.action_type == ActionType.KEYBOARD_HOTKEY:
                success = self._execute_keyboard_hotkey(action)
            elif action.action_type == ActionType.WAIT:
                success = self._execute_wait(action)
            elif action.action_type == ActionType.SCREENSHOT:
                success = True
            elif action.action_type == ActionType.DONE:
                success = True
                message = "Task completed"
            else:
                message = f"Unknown action type: {action.action_type}"

        except Exception as e:
            message = f"Action execution error: {e}"
            logger.error(message)

        # Get screenshot after action
        screenshot_after = None
        if action.action_type != ActionType.DONE:
            screenshot_after = self.backend.screenshot()

        result = ActionResult(
            success=success,
            action=action,
            message=message,
            screenshot_after=screenshot_after,
        )

        # Post-action callback
        if self.on_after_action:
            self.on_after_action(result)

        return result

    def _execute_mouse_move(self, action: ComputerAction) -> bool:
        if not action.mouse:
            return False
        x, y = self.denormalize_coords(action.mouse.x, action.mouse.y)
        return self.backend.move_mouse(x, y, self.config.mouse_move_duration)

    def _execute_mouse_click(self, action: ComputerAction) -> bool:
        if not action.mouse:
            return False
        x, y = self.denormalize_coords(action.mouse.x, action.mouse.y)
        click_type = action.mouse.click_type or ClickType.LEFT
        clicks = 2 if click_type == ClickType.DOUBLE else 1
        button = click_type.value if click_type != ClickType.DOUBLE else "left"
        return self.backend.click(x, y, button=button, clicks=clicks)

    def _execute_mouse_drag(self, action: ComputerAction) -> bool:
        if not action.mouse:
            return False
        start_x, start_y = self.denormalize_coords(action.mouse.x, action.mouse.y)
        end_x = action.mouse.drag_end_x or action.mouse.x
        end_y = action.mouse.drag_end_y or action.mouse.y
        end_x, end_y = self.denormalize_coords(end_x, end_y)
        return self.backend.drag(start_x, start_y, end_x, end_y)

    def _execute_scroll(self, action: ComputerAction) -> bool:
        if not action.scroll:
            return False
        amount = int(action.scroll.amount * 10)  # Scale normalized amount
        return self.backend.scroll(amount, action.scroll.direction)

    def _execute_keyboard_type(self, action: ComputerAction) -> bool:
        if not action.keyboard or not action.keyboard.text:
            return False
        return self.backend.type_text(action.keyboard.text, self.config.typing_interval)

    def _execute_keyboard_key(self, action: ComputerAction) -> bool:
        if not action.keyboard or not action.keyboard.key:
            return False
        return self.backend.press_key(action.keyboard.key, action.keyboard.modifiers)

    def _execute_keyboard_hotkey(self, action: ComputerAction) -> bool:
        if not action.keyboard or not action.keyboard.modifiers:
            return False
        key = action.keyboard.key or ""
        return self.backend.press_key(key, action.keyboard.modifiers)

    def _execute_wait(self, action: ComputerAction) -> bool:
        duration = action.wait_duration or 1.0
        time.sleep(duration)
        return True

    def screenshot(self) -> bytes:
        """Take a screenshot."""
        return self.backend.screenshot()

    def get_mouse_position(self) -> Tuple[float, float]:
        """Get current mouse position (normalized)."""
        x, y = self.backend.get_mouse_position()
        return x / self.screen_width, y / self.screen_height
