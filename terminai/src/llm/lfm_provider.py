#!/usr/bin/env python3
"""
TerminaI - LFM 2.5 (Liquid Foundation Model) Provider

Liquid AI의 LFM 2.5 모델 시리즈 지원:
- LFM2.5-1.2B-Instruct: 텍스트 생성 (에이전트 태스크 최적화)
- LFM2.5-VL-1.6B: 비전-언어 모델

특징:
- 32K 컨텍스트 윈도우
- 28T 토큰 학습
- 다국어 지원 (한국어 포함)
- 함수 호출 지원
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class LFMConfig:
    """LFM 모델 설정."""
    model_id: str = "LiquidAI/LFM2.5-1.2B-Instruct"
    device: str = "auto"
    dtype: str = "bfloat16"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_k: int = 50
    top_p: float = 0.1
    repetition_penalty: float = 1.05
    use_flash_attention: bool = False
    cache_dir: Optional[str] = None


@dataclass
class LFMVLConfig:
    """LFM Vision-Language 모델 설정."""
    model_id: str = "LiquidAI/LFM2.5-VL-1.6B"
    device: str = "auto"
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.1
    min_p: float = 0.15
    repetition_penalty: float = 1.05
    min_image_tokens: int = 64
    max_image_tokens: int = 256
    do_image_splitting: bool = True
    cache_dir: Optional[str] = None


class LFMProvider:
    """
    LFM 2.5 Instruct 모델 프로바이더.

    에이전트 태스크, 데이터 추출, RAG에 최적화됨.
    """

    # 지원 모델
    MODELS = {
        "lfm-instruct": "LiquidAI/LFM2.5-1.2B-Instruct",
        "lfm-base": "LiquidAI/LFM2.5-1.2B-Base",
        "lfm2.5": "LiquidAI/LFM2.5-1.2B-Instruct",
    }

    def __init__(self, config: Optional[LFMConfig] = None):
        self.config = config or LFMConfig()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> bool:
        """모델 로드."""
        if self._loaded:
            return True

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading LFM model: {self.config.model_id}")

            # Dtype 설정
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

            # 모델 로드
            model_kwargs = {
                "device_map": self.config.device,
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }

            if self.config.cache_dir:
                model_kwargs["cache_dir"] = self.config.cache_dir

            if self.config.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
            )

            self.model.eval()
            self._loaded = True

            logger.info(f"LFM model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load LFM model: {e}")
            return False

    def _ensure_loaded(self):
        """모델 로드 확인."""
        if not self._loaded:
            if not self.load():
                raise RuntimeError("Failed to load LFM model")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        텍스트 생성.

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            **kwargs: 추가 생성 파라미터

        Returns:
            생성된 텍스트
        """
        self._ensure_loaded()

        # 메시지 구성
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 토큰화
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)

        # 생성 파라미터
        gen_kwargs = {
            "do_sample": True,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_kwargs)

        # 디코딩 (입력 제외)
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """비동기 텍스트 생성."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, system_prompt, **kwargs)
        )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        스트리밍 텍스트 생성.

        Yields:
            생성된 토큰
        """
        self._ensure_loaded()
        from transformers import TextIteratorStreamer
        from threading import Thread

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            "input_ids": input_ids,
            "do_sample": True,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for text in streamer:
            yield text

        thread.join()

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        함수 호출을 포함한 생성.

        Args:
            prompt: 사용자 프롬프트
            tools: 사용 가능한 도구 목록
            system_prompt: 시스템 프롬프트

        Returns:
            응답 딕셔너리 (텍스트 또는 도구 호출)
        """
        self._ensure_loaded()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 도구가 포함된 채팅 템플릿
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)

        gen_kwargs = {
            "do_sample": True,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
        }

        with torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_kwargs)

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=False,  # 도구 토큰 유지
        )

        # 도구 호출 파싱
        return self._parse_tool_response(response)

    def _parse_tool_response(self, response: str) -> Dict[str, Any]:
        """도구 호출 응답 파싱."""
        import json
        import re

        # 도구 호출 패턴 검색
        tool_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        match = re.search(tool_pattern, response, re.DOTALL)

        if match:
            try:
                tool_call = json.loads(match.group(1))
                return {
                    "type": "tool_call",
                    "tool": tool_call.get("name"),
                    "arguments": tool_call.get("arguments", {}),
                }
            except json.JSONDecodeError:
                pass

        # 일반 텍스트 응답
        clean_response = re.sub(r'<[^>]+>', '', response).strip()
        return {
            "type": "text",
            "content": clean_response,
        }


class LFMVLProvider:
    """
    LFM 2.5 VL (Vision-Language) 모델 프로바이더.

    멀티모달 이해 및 생성을 위한 비전-언어 모델.
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

            logger.info(f"Loading LFM-VL model: {self.config.model_id}")

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

            model_kwargs = {
                "device_map": self.config.device,
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }

            if self.config.cache_dir:
                model_kwargs["cache_dir"] = self.config.cache_dir

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.model_id,
                **model_kwargs
            )

            self.processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
            )

            self.model.eval()
            self._loaded = True

            logger.info("LFM-VL model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load LFM-VL model: {e}")
            return False

    def _ensure_loaded(self):
        """모델 로드 확인."""
        if not self._loaded:
            if not self.load():
                raise RuntimeError("Failed to load LFM-VL model")

    def generate(
        self,
        images: Union[Any, List[Any]],
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        이미지와 텍스트로 생성.

        Args:
            images: PIL Image 또는 이미지 리스트
            prompt: 텍스트 프롬프트
            system_prompt: 시스템 프롬프트

        Returns:
            생성된 텍스트
        """
        self._ensure_loaded()
        from PIL import Image

        # 이미지 리스트로 변환
        if not isinstance(images, list):
            images = [images]

        # 대화 구성
        content = []
        for img in images:
            if isinstance(img, str):
                # URL 또는 경로
                from transformers.image_utils import load_image
                img = load_image(img)
            content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": prompt})

        conversation = [{"role": "user", "content": content}]

        if system_prompt:
            conversation.insert(0, {"role": "system", "content": system_prompt})

        # 입력 처리
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)

        # 생성 파라미터
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "do_sample": True,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
        }

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # 디코딩
        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        return response.strip()

    async def generate_async(
        self,
        images: Union[Any, List[Any]],
        prompt: str,
        **kwargs,
    ) -> str:
        """비동기 생성."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(images, prompt, **kwargs)
        )

    def describe_image(self, image: Any, detail_level: str = "normal") -> str:
        """
        이미지 설명 생성.

        Args:
            image: PIL Image
            detail_level: "brief", "normal", "detailed"

        Returns:
            이미지 설명
        """
        prompts = {
            "brief": "Briefly describe this image in one sentence.",
            "normal": "Describe what you see in this image.",
            "detailed": "Provide a detailed description of this image, including objects, colors, composition, and any text visible.",
        }

        prompt = prompts.get(detail_level, prompts["normal"])
        return self.generate(image, prompt)

    def extract_text(self, image: Any) -> str:
        """
        이미지에서 텍스트 추출 (OCR).

        Args:
            image: PIL Image

        Returns:
            추출된 텍스트
        """
        prompt = "Extract and transcribe all text visible in this image. If there is no text, say 'No text found'."
        return self.generate(image, prompt)

    def analyze_ui(self, screenshot: Any) -> Dict[str, Any]:
        """
        UI 스크린샷 분석.

        Args:
            screenshot: 스크린샷 이미지

        Returns:
            UI 요소 분석 결과
        """
        prompt = """Analyze this UI screenshot and identify:
1. Type of application/website
2. Main UI elements (buttons, text fields, menus, etc.)
3. Any interactive elements with their approximate positions
4. Current state of the interface

Respond in a structured format."""

        response = self.generate(screenshot, prompt)

        return {
            "analysis": response,
            "raw_response": response,
        }

    def locate_element(
        self,
        screenshot: Any,
        element_description: str,
    ) -> Dict[str, Any]:
        """
        UI에서 특정 요소 위치 찾기.

        Args:
            screenshot: 스크린샷
            element_description: 찾을 요소 설명

        Returns:
            요소 위치 정보
        """
        prompt = f"""Look at this screenshot and find: {element_description}

Provide the approximate location of this element:
- Describe where it is on the screen (top/middle/bottom, left/center/right)
- Estimate the normalized coordinates (0-1) for both x and y
- Describe any visual characteristics that help identify it

Format your response as:
LOCATION: [description]
COORDINATES: x=[0-1], y=[0-1]
CONFIDENCE: [low/medium/high]"""

        response = self.generate(screenshot, prompt)

        # 좌표 파싱
        import re
        coords_match = re.search(r'x=([0-9.]+),?\s*y=([0-9.]+)', response)
        x, y = 0.5, 0.5
        if coords_match:
            try:
                x = float(coords_match.group(1))
                y = float(coords_match.group(2))
            except ValueError:
                pass

        return {
            "x": x,
            "y": y,
            "description": response,
            "found": "not found" not in response.lower(),
        }


def create_lfm_provider(
    model_type: str = "instruct",
    **kwargs,
) -> Union[LFMProvider, LFMVLProvider]:
    """
    LFM 프로바이더 팩토리.

    Args:
        model_type: "instruct" 또는 "vl"
        **kwargs: 추가 설정

    Returns:
        LFMProvider 또는 LFMVLProvider
    """
    if model_type.lower() in ["vl", "vision", "multimodal"]:
        config = LFMVLConfig(**kwargs)
        return LFMVLProvider(config)
    else:
        config = LFMConfig(**kwargs)
        return LFMProvider(config)
