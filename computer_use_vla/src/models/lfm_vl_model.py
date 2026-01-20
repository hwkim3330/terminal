#!/usr/bin/env python3
"""
Computer Use VLA - LFM 2.5 VL 모델

Liquid AI의 LFM 2.5 VL (1.6B) 모델을 사용한
비전-언어-액션 모델 구현.

특징:
- 1.6B 파라미터 (경량)
- 네이티브 해상도 처리 (512x512)
- 멀티 이미지 지원
- OCR 및 UI 이해
- 한국어 지원
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

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
class LFMVLConfig:
    """LFM VL 모델 설정."""
    model_id: str = "LiquidAI/LFM2.5-VL-1.6B"
    device: str = "auto"
    dtype: str = "bfloat16"
    max_new_tokens: int = 512
    temperature: float = 0.1
    min_p: float = 0.15
    repetition_penalty: float = 1.05
    min_image_tokens: int = 64
    max_image_tokens: int = 256
    do_image_splitting: bool = True
    screen_width: int = 1920
    screen_height: int = 1080


@dataclass
class LFMVLOutput:
    """LFM VL 모델 출력."""
    action: ComputerAction
    reasoning: str
    ui_elements: List[Dict[str, Any]]
    confidence: float
    raw_response: str


# 시스템 프롬프트
COMPUTER_USE_SYSTEM_PROMPT = """You are an AI agent that controls a computer.

Your task is to:
1. Analyze the screen image
2. Identify UI elements and their locations
3. Decide what action to take
4. Provide precise coordinates for mouse actions

Available actions:
- MOUSE_CLICK: Click at coordinates
- MOUSE_MOVE: Move mouse to coordinates
- MOUSE_DRAG: Drag from one position to another
- KEYBOARD_TYPE: Type text
- KEYBOARD_KEY: Press a key
- SCROLL: Scroll up/down
- WAIT: Wait for a moment
- DONE: Task completed

Always respond in this exact format:
REASONING: [Your step-by-step analysis]
ACTION: [action type]
COORDINATES: [x, y] (normalized 0-1)
CLICK_TYPE: [left/right/double] (if applicable)
TEXT: "[text to type]" (if applicable)
KEY: [key name] (if applicable)
CONFIDENCE: [0-1]"""


class LFMVL:
    """
    LFM 2.5 VL 기반 Computer Use 모델.

    스크린을 분석하고 컴퓨터 제어 액션을 생성합니다.
    """

    def __init__(self, config: Optional[LFMVLConfig] = None):
        self.config = config or LFMVLConfig()
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> bool:
        """모델 로드."""
        if self._loaded:
            return True

        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            logger.info(f"Loading LFM-VL: {self.config.model_id}")

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.model_id,
                device_map=self.config.device,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

            self.processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
            )

            self.model.eval()
            self._loaded = True
            logger.info("LFM-VL loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load LFM-VL: {e}")
            return False

    def _ensure_loaded(self):
        """모델 로드 확인."""
        if not self._loaded:
            if not self.load():
                raise RuntimeError("Failed to load LFM-VL model")

    def _preprocess_image(self, image: Union[Image.Image, str, bytes]) -> Image.Image:
        """이미지 전처리."""
        if isinstance(image, str):
            # 파일 경로
            image = Image.open(image)
        elif isinstance(image, bytes):
            # 바이트 데이터
            import io
            image = Image.open(io.BytesIO(image))

        # RGB로 변환
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def predict(
        self,
        screenshot: Union[Image.Image, str, bytes],
        task: str,
        context: Optional[str] = None,
    ) -> LFMVLOutput:
        """
        스크린샷 분석 및 액션 예측.

        Args:
            screenshot: 스크린샷 이미지
            task: 수행할 태스크
            context: 추가 컨텍스트

        Returns:
            LFMVLOutput
        """
        self._ensure_loaded()

        # 이미지 전처리
        image = self._preprocess_image(screenshot)

        # 프롬프트 생성
        prompt = self._build_prompt(task, context)

        # 대화 구성
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 입력 처리
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                repetition_penalty=self.config.repetition_penalty,
            )

        # 디코딩
        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        # 파싱
        return self._parse_response(response)

    async def predict_async(
        self,
        screenshot: Union[Image.Image, str, bytes],
        task: str,
        context: Optional[str] = None,
    ) -> LFMVLOutput:
        """비동기 예측."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.predict(screenshot, task, context)
        )

    def _build_prompt(self, task: str, context: Optional[str] = None) -> str:
        """프롬프트 생성."""
        prompt = COMPUTER_USE_SYSTEM_PROMPT + f"\n\nTASK: {task}"

        if context:
            prompt += f"\n\nCONTEXT: {context}"

        prompt += "\n\nAnalyze the screenshot and provide your response:"

        return prompt

    def _parse_response(self, response: str) -> LFMVLOutput:
        """응답 파싱."""
        # 기본값
        action_type = ActionType.WAIT
        x, y = 0.5, 0.5
        click_type = None
        text = None
        key = None
        reasoning = ""
        confidence = 0.5

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()

            if line.startswith('REASONING:'):
                reasoning = line[10:].strip()
            elif line.startswith('ACTION:'):
                action_str = line[7:].strip().upper()
                action_map = {
                    'MOUSE_CLICK': ActionType.MOUSE_CLICK,
                    'MOUSE_MOVE': ActionType.MOUSE_MOVE,
                    'MOUSE_DRAG': ActionType.MOUSE_DRAG,
                    'KEYBOARD_TYPE': ActionType.KEYBOARD_TYPE,
                    'KEYBOARD_KEY': ActionType.KEYBOARD_KEY,
                    'SCROLL': ActionType.MOUSE_SCROLL,
                    'WAIT': ActionType.WAIT,
                    'DONE': ActionType.DONE,
                }
                action_type = action_map.get(action_str, ActionType.WAIT)
            elif line.startswith('COORDINATES:'):
                coords = re.findall(r'[\d.]+', line)
                if len(coords) >= 2:
                    x = float(coords[0])
                    y = float(coords[1])
                    # 정규화 확인
                    if x > 1:
                        x = x / self.config.screen_width
                    if y > 1:
                        y = y / self.config.screen_height
            elif line.startswith('CLICK_TYPE:'):
                click_str = line[11:].strip().lower()
                click_map = {
                    'left': ClickType.LEFT,
                    'right': ClickType.RIGHT,
                    'double': ClickType.DOUBLE,
                    'middle': ClickType.MIDDLE,
                }
                click_type = click_map.get(click_str, ClickType.LEFT)
            elif line.startswith('TEXT:'):
                text_match = re.search(r'["\'](.+?)["\']', line)
                if text_match:
                    text = text_match.group(1)
            elif line.startswith('KEY:'):
                key = line[4:].strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line[11:].strip())
                except:
                    confidence = 0.5

        # 액션 생성
        mouse = None
        keyboard = None
        scroll = None

        if action_type in [ActionType.MOUSE_CLICK, ActionType.MOUSE_MOVE, ActionType.MOUSE_DRAG]:
            mouse = MouseAction(
                x=max(0, min(1, x)),
                y=max(0, min(1, y)),
                click_type=click_type or ClickType.LEFT,
            )

        if action_type in [ActionType.KEYBOARD_TYPE, ActionType.KEYBOARD_KEY]:
            keyboard = KeyboardAction(
                text=text,
                key=key,
            )

        if action_type == ActionType.MOUSE_SCROLL:
            scroll = ScrollAction(direction="down", amount=0.3)

        action = ComputerAction(
            action_type=action_type,
            mouse=mouse,
            keyboard=keyboard,
            scroll=scroll,
            reasoning=reasoning,
            confidence=confidence,
        )

        return LFMVLOutput(
            action=action,
            reasoning=reasoning,
            ui_elements=[],
            confidence=confidence,
            raw_response=response,
        )

    def analyze_ui(self, screenshot: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """UI 분석."""
        self._ensure_loaded()

        image = self._preprocess_image(screenshot)

        prompt = """Analyze this UI screenshot and identify:
1. Application type (browser, file manager, terminal, etc.)
2. Main UI elements with their approximate positions
3. Any text or labels visible
4. Interactive elements (buttons, inputs, links)

Respond in a structured format."""

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)

        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        return {"analysis": response}

    def find_element(
        self,
        screenshot: Union[Image.Image, str, bytes],
        element_description: str,
    ) -> Dict[str, Any]:
        """
        UI 요소 찾기.

        Args:
            screenshot: 스크린샷
            element_description: 찾을 요소 설명

        Returns:
            위치 정보
        """
        self._ensure_loaded()

        image = self._preprocess_image(screenshot)

        prompt = f"""Find the UI element: {element_description}

Provide:
1. Whether it exists (yes/no)
2. Its approximate location as normalized coordinates (0-1)
3. Visual description of the element

Format:
FOUND: [yes/no]
COORDINATES: [x, y]
DESCRIPTION: [visual description]"""

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)

        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        # 파싱
        found = "yes" in response.lower()
        x, y = 0.5, 0.5

        coords = re.findall(r'COORDINATES:\s*\[?\s*([\d.]+)\s*,\s*([\d.]+)', response)
        if coords:
            x = float(coords[0][0])
            y = float(coords[0][1])

        return {
            "found": found,
            "x": x,
            "y": y,
            "description": response,
        }

    def extract_text(self, screenshot: Union[Image.Image, str, bytes]) -> str:
        """OCR - 이미지에서 텍스트 추출."""
        self._ensure_loaded()

        image = self._preprocess_image(screenshot)

        prompt = "Extract all visible text from this image. List each text element on a new line."

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)

        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        return response


class LFMVLAgent:
    """
    LFM 2.5 VL 기반 Computer Use 에이전트.

    스크린을 분석하고 컴퓨터를 자율적으로 제어합니다.
    """

    def __init__(self, config: Optional[LFMVLConfig] = None):
        self.config = config or LFMVLConfig()
        self.model = LFMVL(self.config)
        self.controller = None
        self.history: List[Dict[str, Any]] = []

    def _get_controller(self):
        """컨트롤러 lazy load."""
        if self.controller is None:
            from ..controller.computer_controller import ComputerController, ControllerConfig
            self.controller = ComputerController(ControllerConfig(
                screen_width=self.config.screen_width,
                screen_height=self.config.screen_height,
            ))
        return self.controller

    async def run(
        self,
        task: str,
        max_steps: int = 50,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        태스크 실행.

        Args:
            task: 수행할 태스크
            max_steps: 최대 스텝 수
            context: 추가 컨텍스트

        Returns:
            실행 결과
        """
        controller = self._get_controller()
        self.history = []

        logger.info(f"Starting task: {task}")

        for step in range(max_steps):
            # 스크린샷
            screenshot = controller.screenshot()

            # 예측
            output = await self.model.predict_async(screenshot, task, context)

            # 기록
            self.history.append({
                "step": step + 1,
                "action": output.action.to_dict(),
                "reasoning": output.reasoning,
                "confidence": output.confidence,
            })

            # 완료 체크
            if output.action.action_type == ActionType.DONE:
                logger.info(f"Task completed in {step + 1} steps")
                return {
                    "success": True,
                    "steps": step + 1,
                    "history": self.history,
                }

            # 액션 실행
            result = controller.execute(output.action)

            if not result.success:
                logger.warning(f"Action failed: {result.message}")

            # 잠시 대기
            await asyncio.sleep(0.3)

        return {
            "success": False,
            "steps": max_steps,
            "history": self.history,
            "error": "Max steps reached",
        }
