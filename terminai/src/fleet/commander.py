#!/usr/bin/env python3
"""
TerminaI - Fleet Commander

Orchestrates multiple agents across infrastructure.
The "brain" that coordinates distributed agent operations.

Features:
- Multi-agent orchestration
- Task distribution
- Load balancing
- Result aggregation
- Fault tolerance
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime
import uuid

from ..a2a.protocol import (
    A2AProtocol,
    A2AMessage,
    A2ATask,
    AgentCard,
    AgentCapability,
    TaskStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class FleetTask:
    """Task managed by Fleet Commander."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    subtasks: List[A2ATask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agents: Set[str] = field(default_factory=set)
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class FleetConfig:
    """Fleet Commander configuration."""
    max_parallel_tasks: int = 10
    task_timeout: float = 300.0
    retry_count: int = 3
    load_balance: bool = True
    prefer_local: bool = True


class FleetCommander:
    """
    Fleet Commander - orchestrates multiple agents.

    Like a general commanding an army of AI agents,
    the Fleet Commander:
    - Discovers available agents
    - Distributes tasks based on capabilities
    - Monitors progress
    - Handles failures and retries
    - Aggregates results
    """

    def __init__(
        self,
        agent_card: Optional[AgentCard] = None,
        config: Optional[FleetConfig] = None,
    ):
        # Create default agent card if not provided
        self.agent_card = agent_card or AgentCard(
            id=f"fleet-commander-{uuid.uuid4().hex[:8]}",
            name="Fleet Commander",
            description="Orchestrates distributed agent operations",
            capabilities={AgentCapability.COMPUTE},
        )

        self.config = config or FleetConfig()
        self.protocol = A2AProtocol(self.agent_card)

        # Fleet state
        self.fleet: Dict[str, AgentCard] = {}
        self.active_tasks: Dict[str, FleetTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()

        # Local agents (in-process)
        self.local_agents: Dict[str, Any] = {}

        # Stats
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_delegations": 0,
        }

    async def discover(self, endpoints: List[str]) -> int:
        """
        Discover agents at given endpoints.

        Args:
            endpoints: List of agent endpoints to probe

        Returns:
            Number of agents discovered
        """
        discovered = await self.protocol.discover_agents(endpoints)

        for card in discovered:
            self.fleet[card.id] = card
            logger.info(f"Added to fleet: {card.name} with {len(card.capabilities)} capabilities")

        return len(discovered)

    def register_local_agent(self, agent_id: str, agent: Any, capabilities: Set[AgentCapability]):
        """Register a local (in-process) agent."""
        card = AgentCard(
            id=agent_id,
            name=f"Local Agent: {agent_id}",
            capabilities=capabilities,
        )
        self.fleet[agent_id] = card
        self.local_agents[agent_id] = agent
        logger.info(f"Registered local agent: {agent_id}")

    def get_agents_with_capability(self, capability: AgentCapability) -> List[AgentCard]:
        """Find agents with specific capability."""
        return [
            card for card in self.fleet.values()
            if capability in card.capabilities
        ]

    def select_agent(
        self,
        required_capabilities: Optional[Set[AgentCapability]] = None,
        prefer_local: bool = True,
    ) -> Optional[AgentCard]:
        """
        Select best agent for a task.

        Uses load balancing and capability matching.
        """
        candidates = list(self.fleet.values())

        # Filter by capabilities
        if required_capabilities:
            candidates = [
                c for c in candidates
                if required_capabilities.issubset(c.capabilities)
            ]

        if not candidates:
            return None

        # Prefer local agents if configured
        if prefer_local and self.config.prefer_local:
            local = [c for c in candidates if c.id in self.local_agents]
            if local:
                candidates = local

        # Simple round-robin selection
        # TODO: Implement proper load balancing
        return candidates[0]

    async def delegate(
        self,
        agent_id: str,
        task_description: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Delegate task to specific agent.

        Args:
            agent_id: Target agent ID
            task_description: What to do
            input_data: Input data for task

        Returns:
            Task result
        """
        if agent_id not in self.fleet:
            raise ValueError(f"Unknown agent: {agent_id}")

        task = A2ATask(
            description=task_description,
            input_data=input_data or {},
            assigned_agent=agent_id,
        )

        self.stats["total_delegations"] += 1

        # Local agent
        if agent_id in self.local_agents:
            return await self._execute_local(agent_id, task)

        # Remote agent
        return await self._execute_remote(agent_id, task)

    async def _execute_local(self, agent_id: str, task: A2ATask) -> Dict[str, Any]:
        """Execute task on local agent."""
        agent = self.local_agents[agent_id]

        try:
            # Assume agent has a 'run' method
            if hasattr(agent, 'run'):
                result = await agent.run(task.description)
                task.status = TaskStatus.COMPLETED
                task.result = result
                return result
            else:
                raise ValueError(f"Agent {agent_id} has no 'run' method")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            raise

    async def _execute_remote(self, agent_id: str, task: A2ATask) -> Dict[str, Any]:
        """Execute task on remote agent."""
        message = A2AMessage(
            type="request",
            method="execute_task",
            sender=self.agent_card.id,
            receiver=agent_id,
            payload={
                "task": {
                    "description": task.description,
                    "input_data": task.input_data,
                }
            },
        )

        response = await self.protocol.send_message(message)

        if response and "error" not in response.payload:
            task.id = response.payload.get("result", {}).get("task_id", task.id)
            return await self._wait_for_completion(agent_id, task.id)
        else:
            error = response.payload.get("error", "Unknown error") if response else "No response"
            raise RuntimeError(f"Delegation failed: {error}")

    async def _wait_for_completion(
        self,
        agent_id: str,
        task_id: str,
        poll_interval: float = 1.0,
    ) -> Dict[str, Any]:
        """Wait for remote task completion."""
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > self.config.task_timeout:
                raise TimeoutError(f"Task {task_id} timed out")

            # Poll status
            message = A2AMessage(
                type="request",
                method="get_status",
                sender=self.agent_card.id,
                receiver=agent_id,
                payload={"task_id": task_id},
            )

            response = await self.protocol.send_message(message)

            if response:
                task_data = response.payload.get("result", {}).get("task", {})
                status = task_data.get("status")

                if status == TaskStatus.COMPLETED.value:
                    return task_data.get("result", {})
                elif status == TaskStatus.FAILED.value:
                    raise RuntimeError(task_data.get("error", "Task failed"))

            await asyncio.sleep(poll_interval)

    async def execute_parallel(
        self,
        tasks: List[Dict[str, Any]],
        capabilities: Optional[Set[AgentCapability]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tasks in parallel across fleet.

        Args:
            tasks: List of task definitions
            capabilities: Required capabilities

        Returns:
            List of results
        """
        # Find suitable agents
        agents = self.get_agents_with_capability(
            list(capabilities)[0] if capabilities else AgentCapability.SHELL
        )

        if not agents:
            raise ValueError("No agents available with required capabilities")

        # Distribute tasks
        async def execute_task(task_def, agent):
            return await self.delegate(
                agent.id,
                task_def.get("description", ""),
                task_def.get("input_data"),
            )

        # Create task assignments (round-robin)
        assignments = []
        for i, task_def in enumerate(tasks):
            agent = agents[i % len(agents)]
            assignments.append(execute_task(task_def, agent))

        # Execute in parallel
        results = await asyncio.gather(*assignments, return_exceptions=True)

        return results

    async def map_reduce(
        self,
        data: List[Any],
        map_task: str,
        reduce_task: str,
    ) -> Any:
        """
        Perform map-reduce operation across fleet.

        Args:
            data: Data to process
            map_task: Task description for map phase
            reduce_task: Task description for reduce phase

        Returns:
            Reduced result
        """
        # Map phase - distribute data processing
        map_tasks = [
            {"description": map_task, "input_data": {"item": item}}
            for item in data
        ]

        map_results = await self.execute_parallel(map_tasks)

        # Filter out errors
        valid_results = [r for r in map_results if not isinstance(r, Exception)]

        # Reduce phase
        if valid_results:
            reduce_result = await self.delegate(
                self.select_agent().id,
                reduce_task,
                {"results": valid_results},
            )
            return reduce_result

        return None

    def get_fleet_status(self) -> Dict[str, Any]:
        """Get current fleet status."""
        return {
            "total_agents": len(self.fleet),
            "local_agents": len(self.local_agents),
            "remote_agents": len(self.fleet) - len(self.local_agents),
            "active_tasks": len(self.active_tasks),
            "stats": self.stats,
            "agents": [
                {
                    "id": card.id,
                    "name": card.name,
                    "capabilities": [c.name for c in card.capabilities],
                    "is_local": card.id in self.local_agents,
                }
                for card in self.fleet.values()
            ],
        }
