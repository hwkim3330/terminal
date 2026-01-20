# SPDX-License-Identifier: Apache-2.0
# Computer Use VLA - Main Agent
# Autonomous computer control agent

"""
Computer Use Agent

Main agent class that integrates:
- Vision-Language-Action model for understanding and planning
- Controller for executing actions
- Feedback loop for task completion

Inspired by NVIDIA Alpamayo's autonomous driving agent.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image
import io

import torch

from .models.computer_use_vla import (
    ComputerUseVLA,
    ComputerUseVLAConfig,
    ComputerUseOutput,
)
from .controller.computer_controller import (
    ComputerController,
    ControllerConfig,
    ActionResult,
)
from .action_space.computer_action_space import (
    ComputerAction,
    ActionType,
)
from .diffusion.action_flow_matching import MouseTrajectoryGenerator

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the Computer Use Agent."""
    # Model configuration
    model_name: str = "LiquidAI/LFM2.5-VL-1.6B"
    model_dtype: str = "float16"
    device: str = "cuda"

    # Controller configuration
    controller_backend: str = "pyautogui"
    safe_mode: bool = True
    dry_run: bool = False

    # Agent behavior
    max_steps: int = 50
    max_retries: int = 3
    step_delay: float = 0.5  # seconds between steps
    screenshot_delay: float = 0.3  # seconds to wait before screenshot

    # Reasoning
    use_chain_of_thought: bool = True
    verbose: bool = True

    # Screen
    screen_width: int = 1920
    screen_height: int = 1080


@dataclass
class AgentState:
    """Current state of the agent."""
    step_count: int = 0
    task: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    is_running: bool = False
    is_completed: bool = False
    error: Optional[str] = None
    current_screenshot: Optional[bytes] = None


class ComputerUseAgent:
    """
    Autonomous computer use agent.

    Uses a Vision-Language-Action model to understand screen content
    and execute computer actions to complete tasks.

    Example:
        ```python
        agent = ComputerUseAgent()
        agent.run("Open Chrome and search for 'hello world'")
        ```
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the computer use agent."""
        self.config = config or AgentConfig()
        self.state = AgentState()

        # Initialize components
        self._init_model()
        self._init_controller()
        self._init_trajectory_generator()

        # Callbacks
        self.on_step: Optional[Callable[[int, ComputerAction, ActionResult], None]] = None
        self.on_screenshot: Optional[Callable[[bytes], None]] = None
        self.on_reasoning: Optional[Callable[[str], None]] = None

        logger.info("ComputerUseAgent initialized")

    def _init_model(self):
        """Initialize the VLA model."""
        model_config = ComputerUseVLAConfig(
            vlm_name_or_path=self.config.model_name,
            screen_width=self.config.screen_width,
            screen_height=self.config.screen_height,
            use_chain_of_thought=self.config.use_chain_of_thought,
            model_dtype=self.config.model_dtype,
        )

        self.model = ComputerUseVLA(model_config)

        if self.config.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            logger.info("Model loaded on CUDA")
        else:
            logger.info("Model loaded on CPU")

    def _init_controller(self):
        """Initialize the computer controller."""
        controller_config = ControllerConfig(
            screen_width=self.config.screen_width,
            screen_height=self.config.screen_height,
            backend=self.config.controller_backend,
            safe_mode=self.config.safe_mode,
            dry_run=self.config.dry_run,
        )
        self.controller = ComputerController(controller_config)

        # Update screen dimensions from actual screen
        self.config.screen_width = self.controller.screen_width
        self.config.screen_height = self.controller.screen_height

    def _init_trajectory_generator(self):
        """Initialize mouse trajectory generator."""
        self.trajectory_generator = MouseTrajectoryGenerator(
            trajectory_length=16,
            hidden_dim=256,
        )

    def take_screenshot(self) -> Image.Image:
        """Take a screenshot and return as PIL Image."""
        time.sleep(self.config.screenshot_delay)

        screenshot_bytes = self.controller.screenshot()
        self.state.current_screenshot = screenshot_bytes

        if self.on_screenshot:
            self.on_screenshot(screenshot_bytes)

        # Convert to PIL Image
        image = Image.open(io.BytesIO(screenshot_bytes))
        return image

    def run(
        self,
        task: str,
        context: Optional[str] = None,
        initial_screenshot: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent to complete a task.

        Args:
            task: Task description
            context: Optional context information
            initial_screenshot: Optional initial screenshot

        Returns:
            Result dictionary with success status and history
        """
        self.state = AgentState(task=task, is_running=True)

        logger.info(f"Starting task: {task}")

        try:
            while (
                self.state.step_count < self.config.max_steps
                and not self.state.is_completed
                and self.state.is_running
            ):
                result = self._execute_step(context, initial_screenshot)
                initial_screenshot = None  # Only use for first step

                if result.action.action_type == ActionType.DONE:
                    self.state.is_completed = True
                    logger.info("Task completed successfully")
                    break

                time.sleep(self.config.step_delay)

            if self.state.step_count >= self.config.max_steps:
                logger.warning("Max steps reached")

        except KeyboardInterrupt:
            logger.info("Agent interrupted by user")
            self.state.is_running = False
        except Exception as e:
            logger.error(f"Agent error: {e}")
            self.state.error = str(e)
            self.state.is_running = False

        self.state.is_running = False

        return {
            "success": self.state.is_completed,
            "steps": self.state.step_count,
            "history": self.state.history,
            "error": self.state.error,
        }

    def _execute_step(
        self,
        context: Optional[str] = None,
        screenshot: Optional[Image.Image] = None,
    ) -> ActionResult:
        """Execute a single step."""
        self.state.step_count += 1

        if self.config.verbose:
            logger.info(f"Step {self.state.step_count}")

        # Take screenshot if not provided
        if screenshot is None:
            screenshot = self.take_screenshot()

        # Get model prediction
        output = self.model.predict(
            screen_image=screenshot,
            task=self.state.task,
            context=self._build_context(context),
        )

        # Log reasoning
        if self.config.verbose and output.reasoning:
            logger.info(f"Reasoning: {output.reasoning[:200]}...")

        if self.on_reasoning:
            self.on_reasoning(output.reasoning)

        # Execute action
        action = output.action
        result = self.controller.execute(action)

        # Record in history
        self.state.history.append({
            "step": self.state.step_count,
            "action": action.to_dict(),
            "reasoning": output.reasoning,
            "success": result.success,
            "message": result.message,
        })

        # Callback
        if self.on_step:
            self.on_step(self.state.step_count, action, result)

        # Handle failure with retry
        if not result.success:
            logger.warning(f"Action failed: {result.message}")

        return result

    def _build_context(self, user_context: Optional[str] = None) -> str:
        """Build context string from history and user context."""
        parts = []

        if user_context:
            parts.append(user_context)

        # Add recent history summary
        if self.state.history:
            recent = self.state.history[-3:]
            history_str = "\n".join([
                f"Step {h['step']}: {h['action']['action_type']}"
                for h in recent
            ])
            parts.append(f"Recent actions:\n{history_str}")

        return "\n\n".join(parts) if parts else ""

    def stop(self):
        """Stop the agent."""
        self.state.is_running = False
        logger.info("Agent stopped")

    def reset(self):
        """Reset agent state."""
        self.state = AgentState()
        logger.info("Agent reset")

    def step(
        self,
        task: str,
        screenshot: Optional[Image.Image] = None,
    ) -> Tuple[ComputerAction, str]:
        """
        Execute a single step without loop.
        Useful for manual/interactive control.

        Args:
            task: Task description
            screenshot: Optional screenshot

        Returns:
            (action, reasoning) tuple
        """
        self.state.task = task

        if screenshot is None:
            screenshot = self.take_screenshot()

        output = self.model.predict(
            screen_image=screenshot,
            task=task,
        )

        # Execute action
        result = self.controller.execute(output.action)

        return output.action, output.reasoning


class InteractiveAgent(ComputerUseAgent):
    """
    Interactive computer use agent with human-in-the-loop.

    Allows human confirmation before each action.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig()
        config.safe_mode = True
        super().__init__(config)

        self.auto_approve = False

    def _execute_step(
        self,
        context: Optional[str] = None,
        screenshot: Optional[Image.Image] = None,
    ) -> ActionResult:
        """Execute step with confirmation."""
        self.state.step_count += 1

        if screenshot is None:
            screenshot = self.take_screenshot()

        # Get model prediction
        output = self.model.predict(
            screen_image=screenshot,
            task=self.state.task,
            context=self._build_context(context),
        )

        # Show reasoning
        print(f"\n=== Step {self.state.step_count} ===")
        print(f"Reasoning: {output.reasoning}")
        print(f"Proposed action: {output.action.to_dict()}")

        # Get confirmation
        if not self.auto_approve:
            response = input("\nApprove? (y/n/auto/stop): ").strip().lower()

            if response == "stop":
                self.state.is_running = False
                return ActionResult(
                    success=False,
                    action=output.action,
                    message="Stopped by user",
                )
            elif response == "auto":
                self.auto_approve = True
            elif response != "y":
                return ActionResult(
                    success=False,
                    action=output.action,
                    message="Rejected by user",
                )

        # Execute action
        result = self.controller.execute(output.action)

        # Record
        self.state.history.append({
            "step": self.state.step_count,
            "action": output.action.to_dict(),
            "reasoning": output.reasoning,
            "success": result.success,
        })

        return result


def create_agent(
    model: str = "LiquidAI/LFM2.5-VL-1.6B",
    backend: str = "pyautogui",
    safe_mode: bool = True,
    dry_run: bool = False,
    interactive: bool = False,
) -> ComputerUseAgent:
    """
    Factory function to create a computer use agent.

    Args:
        model: VLM model name
        backend: Controller backend ("pyautogui", "xdotool", "dummy")
        safe_mode: Enable safety checks
        dry_run: Don't actually execute actions
        interactive: Use interactive mode with confirmations

    Returns:
        ComputerUseAgent instance
    """
    config = AgentConfig(
        model_name=model,
        controller_backend=backend,
        safe_mode=safe_mode,
        dry_run=dry_run,
    )

    if interactive:
        return InteractiveAgent(config)
    return ComputerUseAgent(config)
