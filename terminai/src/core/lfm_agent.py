#!/usr/bin/env python3
"""
TerminaI - LFM 2.5 기반 에이전트

Liquid AI의 LFM 2.5 모델을 사용하는 OODA Loop 에이전트.

특징:
- LFM 2.5 Instruct (1.2B) - 빠른 추론
- 32K 컨텍스트 윈도우
- 에이전트 태스크 최적화
- 한국어 지원
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..llm.lfm_provider import LFMProvider, LFMConfig, create_lfm_provider
from ..pty.native_pty import NativePTY, PTYSession
from .agent import AgentState, Observation, Orientation, Decision, ActionResult, OODAIteration

logger = logging.getLogger(__name__)


# 시스템 프롬프트 (한국어/영어)
SYSTEM_PROMPT_KO = """당신은 TerminaI, 고급 시스템 운영 AI 에이전트입니다.

역할:
- 시스템 명령어 실행
- 파일 및 디렉토리 관리
- 로그 분석 및 문제 해결
- 자동화 스크립트 작성

규칙:
1. 항상 안전한 명령어만 실행
2. 위험한 작업 전 확인
3. 명확한 추론 과정 제시
4. 오류 발생 시 자동 복구 시도

출력 형식을 정확히 따르세요."""

SYSTEM_PROMPT_EN = """You are TerminaI, an advanced system operations AI agent.

Role:
- Execute system commands
- Manage files and directories
- Analyze logs and troubleshoot
- Write automation scripts

Rules:
1. Only execute safe commands
2. Confirm before dangerous operations
3. Provide clear reasoning
4. Attempt auto-recovery on errors

Follow the output format exactly."""


@dataclass
class LFMAgentConfig:
    """LFM 에이전트 설정."""
    # 모델 설정
    model_id: str = "LiquidAI/LFM2.5-1.2B-Instruct"
    device: str = "auto"
    dtype: str = "bfloat16"

    # 생성 설정
    max_new_tokens: int = 1024
    temperature: float = 0.1

    # 에이전트 설정
    max_iterations: int = 50
    verification_enabled: bool = True
    language: str = "auto"  # "ko", "en", "auto"

    # PTY 설정
    shell: str = "/bin/bash"
    timeout: float = 30.0


class LFMAgent:
    """
    LFM 2.5 기반 OODA Loop 에이전트.

    NVIDIA Alpamayo의 Chain-of-Causation에서 영감을 받은
    재귀적 추론 시스템.
    """

    def __init__(self, config: Optional[LFMAgentConfig] = None):
        self.config = config or LFMAgentConfig()

        # LFM 프로바이더
        self.llm = LFMProvider(LFMConfig(
            model_id=self.config.model_id,
            device=self.config.device,
            dtype=self.config.dtype,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        ))

        # PTY 세션
        self.pty = PTYSession()

        # 상태
        self.state = AgentState.IDLE
        self.history: List[OODAIteration] = []
        self.current_task: Optional[str] = None

        # 시스템 프롬프트
        self.system_prompt = self._get_system_prompt()

        # 콜백
        self.on_state_change: Optional[Callable[[AgentState], None]] = None
        self.on_output: Optional[Callable[[str], None]] = None

    def _get_system_prompt(self) -> str:
        """언어에 따른 시스템 프롬프트."""
        if self.config.language == "ko":
            return SYSTEM_PROMPT_KO
        elif self.config.language == "en":
            return SYSTEM_PROMPT_EN
        else:
            # Auto-detect (default to English for better model performance)
            return SYSTEM_PROMPT_EN

    def _set_state(self, state: AgentState):
        """상태 변경."""
        self.state = state
        if self.on_state_change:
            self.on_state_change(state)
        logger.debug(f"State: {state.name}")

    async def run(self, task: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        태스크 실행.

        Args:
            task: 실행할 태스크
            context: 추가 컨텍스트

        Returns:
            실행 결과
        """
        self.current_task = task
        self.history = []

        logger.info(f"Starting task: {task}")

        # 모델 로드
        if not self.llm._loaded:
            logger.info("Loading LFM model...")
            self.llm.load()

        # PTY 시작
        pty = self.pty.create("main")

        iteration = 0
        while iteration < self.config.max_iterations:
            iteration += 1

            try:
                ooda = await self._ooda_iteration(iteration, pty)
                self.history.append(ooda)

                if self._is_complete(ooda):
                    self._set_state(AgentState.COMPLETED)
                    break

                if self._should_abort(ooda):
                    self._set_state(AgentState.ERROR)
                    break

            except Exception as e:
                logger.error(f"Iteration error: {e}")
                self._set_state(AgentState.ERROR)
                break

        # PTY 정리
        self.pty.close("main")

        return {
            "success": self.state == AgentState.COMPLETED,
            "iterations": iteration,
            "history": [self._serialize(h) for h in self.history],
        }

    async def _ooda_iteration(self, iteration: int, pty: NativePTY) -> OODAIteration:
        """OODA 루프 한 사이클."""

        # 1. OBSERVE
        self._set_state(AgentState.OBSERVING)
        observation = await self._observe(pty)

        # 2. ORIENT
        self._set_state(AgentState.ORIENTING)
        orientation = await self._orient(observation)

        # 3. DECIDE
        self._set_state(AgentState.DECIDING)
        decision = await self._decide(observation, orientation)

        # 4. ACT
        self._set_state(AgentState.ACTING)
        result = await self._act(decision, pty)

        # 5. VERIFY
        if self.config.verification_enabled:
            self._set_state(AgentState.VERIFYING)
            result = await self._verify(decision, result)

        return OODAIteration(
            iteration=iteration,
            observation=observation,
            orientation=orientation,
            decision=decision,
            action_result=result,
        )

    async def _observe(self, pty: NativePTY) -> Observation:
        """관찰 단계."""
        observation = Observation()

        if pty.is_running:
            observation.terminal_output = pty.read_output(timeout=0.3)

        observation.system_state = {
            "iteration": len(self.history) + 1,
            "pty_active": pty.is_running,
        }

        return observation

    async def _orient(self, observation: Observation) -> Orientation:
        """상황 분석 단계."""
        history_summary = self._summarize_history()

        prompt = f"""현재 상황을 분석하세요.

태스크: {self.current_task}

터미널 출력:
```
{observation.terminal_output[-1500:] if observation.terminal_output else "(없음)"}
```

이전 작업:
{history_summary}

다음 형식으로 응답:
SITUATION: [현재 상황 분석]
PROBLEMS: [발견된 문제점, 없으면 "없음"]
NEXT_STEP: [다음에 해야 할 일]
CONFIDENCE: [0.0-1.0]"""

        response = self.llm.generate(prompt, self.system_prompt)
        return self._parse_orientation(response)

    async def _decide(self, observation: Observation, orientation: Orientation) -> Decision:
        """결정 단계."""
        prompt = f"""최적의 액션을 결정하세요.

태스크: {self.current_task}
상황: {orientation.situation_analysis}
문제점: {', '.join(orientation.identified_problems) if orientation.identified_problems else '없음'}

사용 가능한 액션:
- shell: 쉘 명령어 실행 (params: command)
- file_read: 파일 읽기 (params: path)
- file_write: 파일 쓰기 (params: path, content)
- complete: 태스크 완료

다음 형식으로 응답:
ACTION: [액션 타입]
PARAMS: {{"key": "value"}}
REASONING: [이 액션을 선택한 이유]
EXPECTED: [예상 결과]"""

        response = self.llm.generate(prompt, self.system_prompt)
        return self._parse_decision(response)

    async def _act(self, decision: Decision, pty: NativePTY) -> ActionResult:
        """실행 단계."""
        start = time.time()

        try:
            if decision.action_type == "shell":
                cmd = decision.action_params.get("command", "")
                logger.info(f"Executing: {cmd}")

                output = await pty.execute(cmd, timeout=self.config.timeout)

                if self.on_output:
                    self.on_output(output)

                return ActionResult(
                    success=pty.last_exit_code == 0,
                    output=output,
                    duration=time.time() - start,
                )

            elif decision.action_type == "file_read":
                path = decision.action_params.get("path", "")
                from pathlib import Path
                content = Path(path).read_text()
                return ActionResult(success=True, output=content, duration=time.time() - start)

            elif decision.action_type == "file_write":
                path = decision.action_params.get("path", "")
                content = decision.action_params.get("content", "")
                from pathlib import Path
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(content)
                return ActionResult(success=True, output=f"Written to {path}", duration=time.time() - start)

            elif decision.action_type == "complete":
                return ActionResult(success=True, output="Task completed", duration=time.time() - start)

            else:
                return ActionResult(success=False, error=f"Unknown action: {decision.action_type}")

        except Exception as e:
            return ActionResult(success=False, error=str(e), duration=time.time() - start)

    async def _verify(self, decision: Decision, result: ActionResult) -> ActionResult:
        """검증 단계."""
        if not result.success:
            return result

        prompt = f"""액션 결과를 검증하세요.

액션: {decision.action_type}
파라미터: {json.dumps(decision.action_params)}
예상 결과: {decision.expected_outcome}

실제 출력:
```
{result.output[-1000:] if result.output else "(없음)"}
```

결과가 예상과 일치하면 "VERIFIED", 문제가 있으면 "ERROR: [설명]"으로 응답."""

        response = self.llm.generate(prompt, self.system_prompt)

        if "ERROR" in response.upper():
            result.success = False
            result.error = response

        return result

    def _parse_orientation(self, response: str) -> Orientation:
        """상황 분석 파싱."""
        orientation = Orientation()

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('SITUATION:'):
                orientation.situation_analysis = line[10:].strip()
            elif line.startswith('PROBLEMS:'):
                problems = line[9:].strip()
                if problems.lower() not in ['없음', 'none', '']:
                    orientation.identified_problems = [p.strip() for p in problems.split(',')]
            elif line.startswith('CONFIDENCE:'):
                try:
                    orientation.confidence = float(line[11:].strip())
                except:
                    orientation.confidence = 0.5

        return orientation

    def _parse_decision(self, response: str) -> Decision:
        """결정 파싱."""
        decision = Decision()

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('ACTION:'):
                decision.action_type = line[7:].strip().lower()
            elif line.startswith('PARAMS:'):
                try:
                    decision.action_params = json.loads(line[7:].strip())
                except:
                    decision.action_params = {}
            elif line.startswith('REASONING:'):
                decision.reasoning = line[10:].strip()
            elif line.startswith('EXPECTED:'):
                decision.expected_outcome = line[9:].strip()

        return decision

    def _summarize_history(self, max_items: int = 5) -> str:
        """히스토리 요약."""
        if not self.history:
            return "(없음)"

        recent = self.history[-max_items:]
        lines = []
        for h in recent:
            status = "✓" if h.action_result.success else "✗"
            lines.append(f"[{h.iteration}] {status} {h.decision.action_type}")

        return '\n'.join(lines)

    def _is_complete(self, ooda: OODAIteration) -> bool:
        """완료 확인."""
        return ooda.decision.action_type == "complete" and ooda.action_result.success

    def _should_abort(self, ooda: OODAIteration) -> bool:
        """중단 필요 확인."""
        if len(self.history) >= 3:
            recent = self.history[-3:]
            if all(not h.action_result.success for h in recent):
                return True
        return False

    def _serialize(self, ooda: OODAIteration) -> Dict[str, Any]:
        """직렬화."""
        return {
            "iteration": ooda.iteration,
            "action": ooda.decision.action_type,
            "params": ooda.decision.action_params,
            "success": ooda.action_result.success,
            "output": ooda.action_result.output[:500] if ooda.action_result.output else "",
        }


async def run_lfm_agent(task: str, **kwargs) -> Dict[str, Any]:
    """편의 함수: LFM 에이전트 실행."""
    config = LFMAgentConfig(**kwargs)
    agent = LFMAgent(config)
    return await agent.run(task)
