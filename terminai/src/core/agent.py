#!/usr/bin/env python3
"""
TerminaI - Sovereign System Operator
Main Agent Core with OODA Loop Reasoning

Inspired by Gemini CLI architecture with native PTY support.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states."""
    IDLE = auto()
    OBSERVING = auto()
    ORIENTING = auto()
    DECIDING = auto()
    ACTING = auto()
    VERIFYING = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class Observation:
    """Result of observation phase."""
    terminal_output: str = ""
    file_contents: Dict[str, str] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Orientation:
    """Result of orientation phase - understanding context."""
    situation_analysis: str = ""
    identified_problems: List[str] = field(default_factory=list)
    relevant_context: str = ""
    confidence: float = 0.0


@dataclass
class Decision:
    """Result of decision phase."""
    action_type: str = ""
    action_params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    expected_outcome: str = ""
    risk_assessment: str = ""
    confidence: float = 0.0


@dataclass
class ActionResult:
    """Result of action execution."""
    success: bool = False
    output: str = ""
    error: Optional[str] = None
    duration: float = 0.0
    side_effects: List[str] = field(default_factory=list)


@dataclass
class OODAIteration:
    """Single OODA loop iteration."""
    iteration: int
    observation: Observation
    orientation: Orientation
    decision: Decision
    action_result: ActionResult
    timestamp: float = field(default_factory=time.time)


class TerminaIAgent:
    """
    TerminaI - Sovereign System Operator

    Features:
    - OODA Loop: Recursive Observe-Orient-Decide-Act reasoning
    - Native PTY: Real terminal handling with interactive sessions
    - Deep Context: Leverages Gemini 3's massive context window
    - Self-Verification: Validates its own outputs
    - Fleet Commander: Can orchestrate other agents via A2A
    """

    def __init__(
        self,
        llm_provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        max_iterations: int = 50,
        verification_enabled: bool = True,
        verbose: bool = True,
    ):
        self.llm_provider = llm_provider
        self.model = model
        self.max_iterations = max_iterations
        self.verification_enabled = verification_enabled
        self.verbose = verbose

        # State
        self.state = AgentState.IDLE
        self.history: List[OODAIteration] = []
        self.context_buffer: List[str] = []
        self.current_task: Optional[str] = None

        # Components (lazy loaded)
        self._llm = None
        self._pty = None
        self._mcp_client = None
        self._fleet_commander = None

        # Callbacks
        self.on_state_change: Optional[Callable[[AgentState], None]] = None
        self.on_action: Optional[Callable[[Decision], None]] = None
        self.on_output: Optional[Callable[[str], None]] = None

    @property
    def llm(self):
        """Lazy load LLM provider."""
        if self._llm is None:
            from ..llm import create_llm_provider
            self._llm = create_llm_provider(self.llm_provider, self.model)
        return self._llm

    @property
    def pty(self):
        """Lazy load PTY handler."""
        if self._pty is None:
            from ..pty import NativePTY
            self._pty = NativePTY()
        return self._pty

    def _set_state(self, state: AgentState):
        """Update agent state with callback."""
        self.state = state
        if self.on_state_change:
            self.on_state_change(state)
        if self.verbose:
            logger.info(f"State: {state.name}")

    async def run(self, task: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Run OODA loop to complete a task.

        Args:
            task: Task description
            context: Optional additional context

        Returns:
            Result dictionary with success status and history
        """
        self.current_task = task
        self.history = []
        self.context_buffer = [context] if context else []

        logger.info(f"Starting task: {task}")

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            try:
                # Execute OODA loop
                ooda_result = await self._ooda_iteration(iteration)
                self.history.append(ooda_result)

                # Check for task completion
                if self._is_task_complete(ooda_result):
                    self._set_state(AgentState.COMPLETED)
                    logger.info(f"Task completed in {iteration} iterations")
                    break

                # Check for unrecoverable errors
                if self._should_abort(ooda_result):
                    self._set_state(AgentState.ERROR)
                    logger.error("Task aborted due to errors")
                    break

            except Exception as e:
                logger.error(f"OODA iteration failed: {e}")
                self._set_state(AgentState.ERROR)
                break

        return {
            "success": self.state == AgentState.COMPLETED,
            "iterations": iteration,
            "history": [self._serialize_iteration(h) for h in self.history],
            "final_state": self.state.name,
        }

    async def _ooda_iteration(self, iteration: int) -> OODAIteration:
        """Execute single OODA loop iteration."""

        # 1. OBSERVE - Gather information
        self._set_state(AgentState.OBSERVING)
        observation = await self._observe()

        # 2. ORIENT - Understand situation
        self._set_state(AgentState.ORIENTING)
        orientation = await self._orient(observation)

        # 3. DECIDE - Choose action
        self._set_state(AgentState.DECIDING)
        decision = await self._decide(observation, orientation)

        # 4. ACT - Execute action
        self._set_state(AgentState.ACTING)
        action_result = await self._act(decision)

        # 5. VERIFY (optional) - Validate result
        if self.verification_enabled:
            self._set_state(AgentState.VERIFYING)
            action_result = await self._verify(decision, action_result)

        return OODAIteration(
            iteration=iteration,
            observation=observation,
            orientation=orientation,
            decision=decision,
            action_result=action_result,
        )

    async def _observe(self) -> Observation:
        """
        OBSERVE: Gather information about current state.

        - Read terminal output
        - Check file system state
        - Collect system information
        """
        observation = Observation()

        # Get recent terminal output
        if self.pty.is_running:
            observation.terminal_output = self.pty.read_output(timeout=0.5)

        # Get system state
        observation.system_state = {
            "cwd": str(Path.cwd()),
            "pty_running": self.pty.is_running,
            "iteration": len(self.history) + 1,
        }

        return observation

    async def _orient(self, observation: Observation) -> Orientation:
        """
        ORIENT: Analyze the situation and build understanding.

        Uses LLM to:
        - Understand current state
        - Identify problems
        - Build relevant context
        """
        # Build prompt for orientation
        prompt = self._build_orientation_prompt(observation)

        # Get LLM analysis
        response = await self.llm.generate(prompt)

        # Parse orientation from response
        return self._parse_orientation(response)

    async def _decide(self, observation: Observation, orientation: Orientation) -> Decision:
        """
        DECIDE: Choose the best action to take.

        Uses LLM to:
        - Generate action options
        - Evaluate risks
        - Select optimal action
        """
        # Build prompt for decision
        prompt = self._build_decision_prompt(observation, orientation)

        # Get LLM decision
        response = await self.llm.generate(prompt)

        # Parse decision from response
        decision = self._parse_decision(response)

        if self.on_action:
            self.on_action(decision)

        return decision

    async def _act(self, decision: Decision) -> ActionResult:
        """
        ACT: Execute the decided action.

        Supports:
        - Shell commands via PTY
        - File operations
        - Tool invocations (MCP)
        - Agent orchestration (A2A)
        """
        start_time = time.time()

        try:
            if decision.action_type == "shell":
                result = await self._execute_shell(decision.action_params)
            elif decision.action_type == "file_read":
                result = await self._execute_file_read(decision.action_params)
            elif decision.action_type == "file_write":
                result = await self._execute_file_write(decision.action_params)
            elif decision.action_type == "mcp_tool":
                result = await self._execute_mcp_tool(decision.action_params)
            elif decision.action_type == "delegate":
                result = await self._execute_delegation(decision.action_params)
            elif decision.action_type == "complete":
                result = ActionResult(success=True, output="Task marked complete")
            else:
                result = ActionResult(success=False, error=f"Unknown action: {decision.action_type}")

        except Exception as e:
            result = ActionResult(success=False, error=str(e))

        result.duration = time.time() - start_time

        if self.on_output:
            self.on_output(result.output)

        return result

    async def _verify(self, decision: Decision, result: ActionResult) -> ActionResult:
        """
        VERIFY: Validate the action result.

        Self-checks:
        - Did the action succeed?
        - Does the output match expectations?
        - Are there any errors to catch?
        """
        if not result.success:
            return result

        # Build verification prompt
        prompt = self._build_verification_prompt(decision, result)

        # Get LLM verification
        response = await self.llm.generate(prompt)

        # Check if verification passed
        if "VERIFIED" in response.upper():
            return result
        elif "ERROR" in response.upper() or "FAILED" in response.upper():
            # Extract error from response
            result.success = False
            result.error = f"Verification failed: {response[:200]}"

        return result

    async def _execute_shell(self, params: Dict[str, Any]) -> ActionResult:
        """Execute shell command via PTY."""
        command = params.get("command", "")
        timeout = params.get("timeout", 30)
        interactive = params.get("interactive", False)

        if not command:
            return ActionResult(success=False, error="No command provided")

        logger.info(f"Executing: {command}")

        if interactive:
            # Use PTY for interactive commands
            output = await self.pty.execute_interactive(command, timeout=timeout)
        else:
            # Simple command execution
            output = await self.pty.execute(command, timeout=timeout)

        success = self.pty.last_exit_code == 0

        return ActionResult(
            success=success,
            output=output,
            error=None if success else f"Exit code: {self.pty.last_exit_code}",
        )

    async def _execute_file_read(self, params: Dict[str, Any]) -> ActionResult:
        """Read file contents."""
        path = params.get("path", "")

        try:
            content = Path(path).read_text()
            return ActionResult(success=True, output=content)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def _execute_file_write(self, params: Dict[str, Any]) -> ActionResult:
        """Write file contents."""
        path = params.get("path", "")
        content = params.get("content", "")

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content)
            return ActionResult(success=True, output=f"Written to {path}")
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def _execute_mcp_tool(self, params: Dict[str, Any]) -> ActionResult:
        """Execute MCP tool."""
        if self._mcp_client is None:
            from ..mcp import MCPClient
            self._mcp_client = MCPClient()

        tool_name = params.get("tool", "")
        tool_params = params.get("params", {})

        try:
            result = await self._mcp_client.call_tool(tool_name, tool_params)
            return ActionResult(success=True, output=json.dumps(result))
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def _execute_delegation(self, params: Dict[str, Any]) -> ActionResult:
        """Delegate task to another agent via A2A."""
        if self._fleet_commander is None:
            from ..fleet import FleetCommander
            self._fleet_commander = FleetCommander()

        agent_id = params.get("agent_id", "")
        task = params.get("task", "")

        try:
            result = await self._fleet_commander.delegate(agent_id, task)
            return ActionResult(success=True, output=json.dumps(result))
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def _build_orientation_prompt(self, observation: Observation) -> str:
        """Build prompt for orientation phase."""
        history_context = self._get_history_context()

        return f"""You are TerminaI, a Sovereign System Operator.

TASK: {self.current_task}

CURRENT OBSERVATION:
Terminal Output:
```
{observation.terminal_output[-2000:] if observation.terminal_output else "(no output)"}
```

System State:
{json.dumps(observation.system_state, indent=2)}

HISTORY:
{history_context}

Analyze the current situation:
1. What is the current state?
2. What problems or blockers exist?
3. What is the relevant context for the next action?
4. How confident are you in your understanding? (0-1)

Respond in this format:
SITUATION: <your analysis>
PROBLEMS: <list of problems, or "none">
CONTEXT: <relevant context>
CONFIDENCE: <0.0 to 1.0>
"""

    def _build_decision_prompt(self, observation: Observation, orientation: Orientation) -> str:
        """Build prompt for decision phase."""
        return f"""You are TerminaI, a Sovereign System Operator.

TASK: {self.current_task}

SITUATION ANALYSIS:
{orientation.situation_analysis}

IDENTIFIED PROBLEMS:
{', '.join(orientation.identified_problems) if orientation.identified_problems else 'None'}

AVAILABLE ACTIONS:
- shell: Execute a shell command (params: command, timeout, interactive)
- file_read: Read a file (params: path)
- file_write: Write to a file (params: path, content)
- mcp_tool: Call an MCP tool (params: tool, params)
- delegate: Delegate to another agent (params: agent_id, task)
- complete: Mark task as complete

Choose the best action to make progress on the task.

Respond in this format:
ACTION_TYPE: <action type>
PARAMS: <JSON params>
REASONING: <why this action>
EXPECTED_OUTCOME: <what should happen>
RISK: <potential risks>
CONFIDENCE: <0.0 to 1.0>
"""

    def _build_verification_prompt(self, decision: Decision, result: ActionResult) -> str:
        """Build prompt for verification phase."""
        return f"""You are TerminaI, verifying an action result.

ACTION: {decision.action_type}
PARAMS: {json.dumps(decision.action_params)}
EXPECTED: {decision.expected_outcome}

ACTUAL OUTPUT:
```
{result.output[-2000:] if result.output else "(no output)"}
```

Did the action achieve its expected outcome?
Are there any errors or unexpected results?

Respond with either:
- VERIFIED: <brief explanation>
- ERROR: <what went wrong>
"""

    def _parse_orientation(self, response: str) -> Orientation:
        """Parse orientation from LLM response."""
        orientation = Orientation()

        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith('SITUATION:'):
                orientation.situation_analysis = line[10:].strip()
            elif line.startswith('PROBLEMS:'):
                problems = line[9:].strip()
                if problems.lower() != 'none':
                    orientation.identified_problems = [p.strip() for p in problems.split(',')]
            elif line.startswith('CONTEXT:'):
                orientation.relevant_context = line[8:].strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    orientation.confidence = float(line[11:].strip())
                except:
                    orientation.confidence = 0.5

        return orientation

    def _parse_decision(self, response: str) -> Decision:
        """Parse decision from LLM response."""
        decision = Decision()

        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith('ACTION_TYPE:'):
                decision.action_type = line[12:].strip().lower()
            elif line.startswith('PARAMS:'):
                try:
                    decision.action_params = json.loads(line[7:].strip())
                except:
                    decision.action_params = {}
            elif line.startswith('REASONING:'):
                decision.reasoning = line[10:].strip()
            elif line.startswith('EXPECTED_OUTCOME:'):
                decision.expected_outcome = line[17:].strip()
            elif line.startswith('RISK:'):
                decision.risk_assessment = line[5:].strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    decision.confidence = float(line[11:].strip())
                except:
                    decision.confidence = 0.5

        return decision

    def _get_history_context(self, max_items: int = 5) -> str:
        """Get recent history as context."""
        if not self.history:
            return "(no history)"

        recent = self.history[-max_items:]
        context_parts = []

        for h in recent:
            context_parts.append(
                f"[{h.iteration}] {h.decision.action_type}: "
                f"{'SUCCESS' if h.action_result.success else 'FAILED'}"
            )

        return '\n'.join(context_parts)

    def _is_task_complete(self, ooda: OODAIteration) -> bool:
        """Check if task is complete."""
        return ooda.decision.action_type == "complete" and ooda.action_result.success

    def _should_abort(self, ooda: OODAIteration) -> bool:
        """Check if we should abort."""
        # Abort on repeated failures
        if len(self.history) >= 3:
            recent = self.history[-3:]
            if all(not h.action_result.success for h in recent):
                return True
        return False

    def _serialize_iteration(self, ooda: OODAIteration) -> Dict[str, Any]:
        """Serialize OODA iteration for output."""
        return {
            "iteration": ooda.iteration,
            "action_type": ooda.decision.action_type,
            "action_params": ooda.decision.action_params,
            "reasoning": ooda.decision.reasoning,
            "success": ooda.action_result.success,
            "output": ooda.action_result.output[:500],
            "error": ooda.action_result.error,
            "timestamp": ooda.timestamp,
        }
