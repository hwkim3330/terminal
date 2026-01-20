# SPDX-License-Identifier: Apache-2.0
# Computer Use VLA - Main Model
# Inspired by NVIDIA Alpamayo ReasoningVLA architecture

"""
Computer Use Vision-Language-Action Model

Architecture based on Alpamayo R1:
- Vision encoder for screen understanding
- Language model for reasoning (Chain-of-Thought)
- Action decoder for computer control
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoProcessor,
    AutoModelForVision2Seq,
)

from ..action_space.computer_action_space import (
    ComputerAction,
    ComputerActionSpace,
    ContinuousComputerActionSpace,
    ActionType,
    MouseAction,
    KeyboardAction,
    ClickType,
)

logger = logging.getLogger(__name__)


# Special tokens for computer use VLA
SPECIAL_TOKENS = {
    "screen_start": "<|screen|>",
    "screen_end": "<|/screen|>",
    "action_start": "<|action|>",
    "action_end": "<|/action|>",
    "reasoning_start": "<|reasoning|>",
    "reasoning_end": "<|/reasoning|>",
    "mouse_move": "<|mouse_move|>",
    "mouse_click": "<|mouse_click|>",
    "mouse_drag": "<|mouse_drag|>",
    "keyboard_type": "<|keyboard_type|>",
    "keyboard_key": "<|keyboard_key|>",
    "scroll": "<|scroll|>",
    "wait": "<|wait|>",
    "done": "<|done|>",
    "ui_element": "<|ui|>",
    "coordinates": "<|coords|>",
}


@dataclass
class ComputerUseOutput:
    """Output from Computer Use VLA model."""
    action: ComputerAction
    reasoning: str
    ui_elements: Optional[List[Dict[str, Any]]] = None
    raw_output: Optional[str] = None
    logits: Optional[torch.Tensor] = None


class ComputerUseVLAConfig(PretrainedConfig):
    """Configuration for Computer Use VLA model."""

    model_type = "computer_use_vla"

    def __init__(
        self,
        vlm_name_or_path: str = "LiquidAI/LFM2.5-VL-1.6B",
        vlm_backend: str = "lfm",  # "lfm" or "qwen"
        screen_width: int = 1920,
        screen_height: int = 1080,
        action_space_type: str = "continuous",  # "continuous" or "discrete"
        grid_size: int = 100,
        max_reasoning_tokens: int = 512,
        max_action_tokens: int = 128,
        use_chain_of_thought: bool = True,
        use_ui_detection: bool = True,
        model_dtype: str = "float16",
        attn_implementation: str = "flash_attention_2",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vlm_name_or_path = vlm_name_or_path
        self.vlm_backend = vlm_backend.lower()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.action_space_type = action_space_type
        self.grid_size = grid_size
        self.max_reasoning_tokens = max_reasoning_tokens
        self.max_action_tokens = max_action_tokens
        self.use_chain_of_thought = use_chain_of_thought
        self.use_ui_detection = use_ui_detection
        self.model_dtype = model_dtype
        self.attn_implementation = attn_implementation


class ChainOfThoughtReasoning(nn.Module):
    """
    Chain-of-Thought reasoning module for computer use.
    Inspired by Alpamayo's Chain-of-Causation.

    Reasoning steps:
    1. Screen Analysis - What is on the screen?
    2. Task Understanding - What needs to be done?
    3. UI Element Identification - Where are the relevant elements?
    4. Action Planning - What action should be taken?
    5. Safety Check - Is this action safe?
    """

    REASONING_TEMPLATE = """<|reasoning|>
**Screen Analysis:**
{screen_analysis}

**Task Understanding:**
{task_understanding}

**UI Elements:**
{ui_elements}

**Action Plan:**
{action_plan}

**Safety Check:**
{safety_check}
<|/reasoning|>"""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

        # Reasoning step classifiers
        self.screen_analyzer = nn.Linear(hidden_size, hidden_size)
        self.task_understander = nn.Linear(hidden_size, hidden_size)
        self.ui_detector = nn.Linear(hidden_size, hidden_size)
        self.action_planner = nn.Linear(hidden_size, hidden_size)
        self.safety_checker = nn.Linear(hidden_size, 2)  # safe/unsafe

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process hidden states through reasoning chain.

        Args:
            hidden_states: [B, seq_len, hidden_size]
            attention_mask: [B, seq_len]

        Returns:
            reasoning_output: [B, hidden_size]
            reasoning_steps: Dict of intermediate outputs
        """
        # Pool hidden states
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = hidden_states.mean(dim=1)

        # Apply reasoning steps
        screen_features = torch.relu(self.screen_analyzer(pooled))
        task_features = torch.relu(self.task_understander(screen_features))
        ui_features = torch.relu(self.ui_detector(task_features))
        action_features = torch.relu(self.action_planner(ui_features))
        safety_logits = self.safety_checker(action_features)

        reasoning_steps = {
            "screen_features": screen_features,
            "task_features": task_features,
            "ui_features": ui_features,
            "action_features": action_features,
            "safety_logits": safety_logits,
        }

        return action_features, reasoning_steps


class ActionDecoder(nn.Module):
    """
    Action decoder using flow matching for smooth trajectories.
    Based on Alpamayo's diffusion-based trajectory decoder.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        action_dims: int = 10,
        num_diffusion_steps: int = 10,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dims = action_dims
        self.num_diffusion_steps = num_diffusion_steps

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dims),
        )

        # Action type classifier
        self.action_type_head = nn.Linear(hidden_size, len(ActionType))

        # Coordinate regression
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # x, y, end_x, end_y
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        reasoning_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode action from hidden states.

        Args:
            hidden_states: [B, hidden_size]
            reasoning_features: [B, hidden_size] optional reasoning context

        Returns:
            action_tensor: [B, action_dims]
            action_type_logits: [B, num_action_types]
            coordinates: [B, 4] (x, y, end_x, end_y)
        """
        if reasoning_features is not None:
            combined = hidden_states + reasoning_features
        else:
            combined = hidden_states

        action_tensor = self.action_head(combined)
        action_type_logits = self.action_type_head(combined)
        coordinates = self.coord_head(combined)

        return action_tensor, action_type_logits, coordinates


class ComputerUseVLA(PreTrainedModel):
    """
    Computer Use Vision-Language-Action Model.

    Combines:
    - Vision encoder (from VLM) for screen understanding
    - Language model for Chain-of-Thought reasoning
    - Action decoder for computer control output

    Supports multiple VLM backends:
    - LFM 2.5 VL (Liquid AI) - lightweight, edge-friendly
    - Qwen-VL - more powerful but larger
    """

    config_class = ComputerUseVLAConfig
    base_model_prefix = "vlm"

    def __init__(self, config: ComputerUseVLAConfig):
        super().__init__(config)

        self.config = config

        # Initialize VLM backbone
        self._initialize_vlm(config)

        # Initialize action space
        if config.action_space_type == "continuous":
            self.action_space = ContinuousComputerActionSpace(
                screen_width=config.screen_width,
                screen_height=config.screen_height,
            )
        else:
            from ..action_space.computer_action_space import DiscreteComputerActionSpace
            self.action_space = DiscreteComputerActionSpace(
                screen_width=config.screen_width,
                screen_height=config.screen_height,
                grid_size=config.grid_size,
            )

        # Get hidden size from VLM
        hidden_size = self._get_hidden_size()

        # Chain-of-Thought reasoning
        if config.use_chain_of_thought:
            self.reasoning = ChainOfThoughtReasoning(hidden_size)

        # Action decoder
        self.action_decoder = ActionDecoder(
            hidden_size=hidden_size,
            action_dims=self.action_space.get_action_space_dims()[0],
        )

        # Special token embeddings
        self.special_token_ids = {}

        logger.info(f"ComputerUseVLA initialized with {config.vlm_name_or_path}")

    def _initialize_vlm(self, config: ComputerUseVLAConfig):
        """Initialize Vision-Language Model backbone."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_dtype = dtype_map.get(config.model_dtype, torch.float16)

        try:
            self.processor = AutoProcessor.from_pretrained(
                config.vlm_name_or_path,
                trust_remote_code=True,
            )

            self.vlm = AutoModelForVision2Seq.from_pretrained(
                config.vlm_name_or_path,
                torch_dtype=model_dtype,
                trust_remote_code=True,
                attn_implementation=config.attn_implementation if config.attn_implementation != "flash_attention_2" else None,
            )

            # Add special tokens
            special_tokens_list = list(SPECIAL_TOKENS.values())
            self.processor.tokenizer.add_tokens(special_tokens_list, special_tokens=True)

            # Resize embeddings
            self.vlm.resize_token_embeddings(len(self.processor.tokenizer))

            # Store special token ids
            self.special_token_ids = {
                k: self.processor.tokenizer.convert_tokens_to_ids(v)
                for k, v in SPECIAL_TOKENS.items()
            }

        except Exception as e:
            logger.warning(f"Failed to load VLM: {e}. Using dummy mode.")
            self.vlm = None
            self.processor = None

    def _get_hidden_size(self) -> int:
        """Get hidden size from VLM config."""
        if self.vlm is None:
            return 768
        if hasattr(self.vlm.config, "hidden_size"):
            return self.vlm.config.hidden_size
        if hasattr(self.vlm.config, "text_config"):
            return self.vlm.config.text_config.hidden_size
        return 768

    def create_prompt(
        self,
        task: str,
        context: Optional[str] = None,
    ) -> str:
        """Create prompt for computer use task."""
        prompt = f"""You are an AI agent that controls a computer. Analyze the screen and perform the requested task.

Task: {task}
"""
        if context:
            prompt += f"\nContext: {context}\n"

        prompt += """
Analyze the screen step by step:
1. What UI elements are visible?
2. Where is the relevant element for this task?
3. What action should be taken?
4. Provide the action with precise coordinates.

Output format:
<|reasoning|>
Your step-by-step analysis
<|/reasoning|>
<|action|>
ACTION_TYPE: [mouse_click/mouse_move/keyboard_type/scroll/wait/done]
COORDINATES: [x, y] (normalized 0-1)
CLICK_TYPE: [left/right/double] (if applicable)
TEXT: "text to type" (if applicable)
<|/action|>
"""
        return prompt

    @torch.no_grad()
    def predict(
        self,
        screen_image: torch.Tensor,
        task: str,
        context: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> ComputerUseOutput:
        """
        Predict action from screen image and task.

        Args:
            screen_image: Screen capture as tensor [C, H, W] or PIL Image
            task: Task description
            context: Optional context information
            max_new_tokens: Maximum tokens to generate

        Returns:
            ComputerUseOutput with action and reasoning
        """
        if self.vlm is None:
            return self._dummy_predict(task)

        # Create prompt
        prompt = self.create_prompt(task, context)

        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=screen_image,
            return_tensors="pt",
        ).to(self.vlm.device)

        # Generate response
        outputs = self.vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Decode response
        response = self.processor.decode(outputs.sequences[0], skip_special_tokens=False)

        # Parse response to extract action
        action, reasoning = self._parse_response(response)

        return ComputerUseOutput(
            action=action,
            reasoning=reasoning,
            raw_output=response,
        )

    def _parse_response(self, response: str) -> Tuple[ComputerAction, str]:
        """Parse model response to extract action and reasoning."""
        import re

        # Extract reasoning
        reasoning_match = re.search(
            r'<\|reasoning\|>(.*?)<\|/reasoning\|>',
            response,
            re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract action
        action_match = re.search(
            r'<\|action\|>(.*?)<\|/action\|>',
            response,
            re.DOTALL
        )

        if action_match:
            action_text = action_match.group(1)
            action = self._parse_action_text(action_text)
        else:
            # Default action
            action = ComputerAction(
                action_type=ActionType.WAIT,
                wait_duration=1.0,
                reasoning=reasoning,
            )

        action.reasoning = reasoning
        return action, reasoning

    def _parse_action_text(self, action_text: str) -> ComputerAction:
        """Parse action text to ComputerAction."""
        import re

        # Extract action type
        action_type_match = re.search(r'ACTION_TYPE:\s*(\w+)', action_text, re.IGNORECASE)
        action_type_str = action_type_match.group(1).lower() if action_type_match else "wait"

        action_type_map = {
            "mouse_click": ActionType.MOUSE_CLICK,
            "mouse_move": ActionType.MOUSE_MOVE,
            "mouse_drag": ActionType.MOUSE_DRAG,
            "keyboard_type": ActionType.KEYBOARD_TYPE,
            "keyboard_key": ActionType.KEYBOARD_KEY,
            "scroll": ActionType.MOUSE_SCROLL,
            "wait": ActionType.WAIT,
            "done": ActionType.DONE,
        }
        action_type = action_type_map.get(action_type_str, ActionType.WAIT)

        # Extract coordinates
        coords_match = re.search(r'COORDINATES:\s*\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?', action_text)
        x, y = 0.5, 0.5
        if coords_match:
            x = float(coords_match.group(1))
            y = float(coords_match.group(2))
            # Ensure normalized
            if x > 1:
                x = x / self.config.screen_width
            if y > 1:
                y = y / self.config.screen_height

        # Extract click type
        click_match = re.search(r'CLICK_TYPE:\s*(\w+)', action_text, re.IGNORECASE)
        click_type = None
        if click_match:
            click_str = click_match.group(1).lower()
            click_map = {"left": ClickType.LEFT, "right": ClickType.RIGHT, "double": ClickType.DOUBLE}
            click_type = click_map.get(click_str, ClickType.LEFT)

        # Extract text
        text_match = re.search(r'TEXT:\s*["\'](.+?)["\']', action_text)
        text = text_match.group(1) if text_match else None

        # Build action
        mouse = None
        keyboard = None

        if action_type in [ActionType.MOUSE_CLICK, ActionType.MOUSE_MOVE, ActionType.MOUSE_DRAG]:
            mouse = MouseAction(x=x, y=y, click_type=click_type)

        if action_type in [ActionType.KEYBOARD_TYPE, ActionType.KEYBOARD_KEY]:
            keyboard = KeyboardAction(text=text)

        return ComputerAction(
            action_type=action_type,
            mouse=mouse,
            keyboard=keyboard,
            confidence=1.0,
        )

    def _dummy_predict(self, task: str) -> ComputerUseOutput:
        """Dummy prediction for testing without model."""
        reasoning = f"[DUMMY MODE] Task: {task}. Moving to center of screen."
        action = ComputerAction(
            action_type=ActionType.MOUSE_MOVE,
            mouse=MouseAction(x=0.5, y=0.5),
            reasoning=reasoning,
            confidence=0.5,
        )
        return ComputerUseOutput(
            action=action,
            reasoning=reasoning,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            pixel_values: Image tensor [B, C, H, W]
            labels: Target token IDs [B, seq_len]

        Returns:
            Dict with loss and logits
        """
        if self.vlm is None:
            return {"loss": torch.tensor(0.0)}

        # VLM forward pass
        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        result = {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

        # Apply Chain-of-Thought reasoning if enabled
        if self.config.use_chain_of_thought and hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1]
            reasoning_output, reasoning_steps = self.reasoning(hidden_states, attention_mask)
            result["reasoning_features"] = reasoning_output
            result["safety_logits"] = reasoning_steps["safety_logits"]

        return result
