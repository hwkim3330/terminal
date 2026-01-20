#!/usr/bin/env python3
"""
TerminaI - A2A (Agent-to-Agent) Protocol

Enables communication and orchestration between multiple agents.
Based on Google's A2A protocol specification.

Features:
- Agent discovery
- Task delegation
- Status monitoring
- Result aggregation
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Agent capabilities."""
    SHELL = auto()
    FILE_OPS = auto()
    WEB_BROWSE = auto()
    CODE_EDIT = auto()
    DATABASE = auto()
    API_CALLS = auto()
    COMPUTE = auto()


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCard:
    """
    Agent Card - describes an agent's identity and capabilities.

    Similar to business card for agents.
    """
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    capabilities: Set[AgentCapability] = field(default_factory=set)
    endpoint: Optional[str] = None  # HTTP endpoint for remote agents
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": [c.name for c in self.capabilities],
            "endpoint": self.endpoint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            capabilities={AgentCapability[c] for c in data.get("capabilities", [])},
            endpoint=data.get("endpoint"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class A2ATask:
    """Task to be executed by an agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    parent_task: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "input_data": self.input_data,
            "assigned_agent": self.assigned_agent,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "parent_task": self.parent_task,
            "subtasks": self.subtasks,
        }


@dataclass
class A2AMessage:
    """Message between agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""  # request, response, notification
    method: str = ""  # execute_task, get_status, cancel_task, etc.
    sender: str = ""
    receiver: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "method": self.method,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=data.get("type", ""),
            method=data.get("method", ""),
            sender=data.get("sender", ""),
            receiver=data.get("receiver", ""),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
        )


class A2AProtocol:
    """
    A2A Protocol implementation.

    Handles agent-to-agent communication including:
    - Discovery
    - Task delegation
    - Status updates
    - Result collection
    """

    def __init__(self, agent_card: AgentCard):
        self.agent_card = agent_card
        self.known_agents: Dict[str, AgentCard] = {}
        self.pending_tasks: Dict[str, A2ATask] = {}
        self.message_handlers: Dict[str, Callable] = {}

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default message handlers."""
        self.message_handlers["discover"] = self._handle_discover
        self.message_handlers["execute_task"] = self._handle_execute_task
        self.message_handlers["get_status"] = self._handle_get_status
        self.message_handlers["cancel_task"] = self._handle_cancel_task
        self.message_handlers["ping"] = self._handle_ping

    async def _handle_discover(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle discovery request."""
        return {"agent_card": self.agent_card.to_dict()}

    async def _handle_execute_task(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle task execution request."""
        task_data = message.payload.get("task", {})
        task = A2ATask(
            description=task_data.get("description", ""),
            input_data=task_data.get("input_data", {}),
            assigned_agent=self.agent_card.id,
        )

        self.pending_tasks[task.id] = task

        return {"task_id": task.id, "status": "accepted"}

    async def _handle_get_status(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle status request."""
        task_id = message.payload.get("task_id")
        if task_id in self.pending_tasks:
            return {"task": self.pending_tasks[task_id].to_dict()}
        return {"error": "Task not found"}

    async def _handle_cancel_task(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle task cancellation."""
        task_id = message.payload.get("task_id")
        if task_id in self.pending_tasks:
            self.pending_tasks[task_id].status = TaskStatus.CANCELLED
            return {"status": "cancelled"}
        return {"error": "Task not found"}

    async def _handle_ping(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle ping request."""
        return {"pong": True, "agent_id": self.agent_card.id}

    async def handle_message(self, message: A2AMessage) -> A2AMessage:
        """Handle incoming A2A message."""
        handler = self.message_handlers.get(message.method)

        if handler:
            try:
                result = await handler(message)
                return A2AMessage(
                    type="response",
                    method=message.method,
                    sender=self.agent_card.id,
                    receiver=message.sender,
                    payload={"result": result},
                )
            except Exception as e:
                return A2AMessage(
                    type="response",
                    method=message.method,
                    sender=self.agent_card.id,
                    receiver=message.sender,
                    payload={"error": str(e)},
                )
        else:
            return A2AMessage(
                type="response",
                method=message.method,
                sender=self.agent_card.id,
                receiver=message.sender,
                payload={"error": f"Unknown method: {message.method}"},
            )

    def register_agent(self, agent_card: AgentCard):
        """Register a known agent."""
        self.known_agents[agent_card.id] = agent_card
        logger.info(f"Registered agent: {agent_card.name} ({agent_card.id})")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.known_agents:
            del self.known_agents[agent_id]

    async def send_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Send message to another agent."""
        target = self.known_agents.get(message.receiver)

        if not target:
            logger.error(f"Unknown agent: {message.receiver}")
            return None

        if target.endpoint:
            # Remote agent - send via HTTP
            return await self._send_http(target.endpoint, message)
        else:
            # Local agent - direct call
            logger.warning("Local agent communication not implemented")
            return None

    async def _send_http(self, endpoint: str, message: A2AMessage) -> Optional[A2AMessage]:
        """Send message via HTTP."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{endpoint}/a2a",
                    json=message.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return A2AMessage.from_dict(data)

        except Exception as e:
            logger.error(f"HTTP send error: {e}")

        return None

    async def discover_agents(self, endpoints: List[str]) -> List[AgentCard]:
        """Discover agents at given endpoints."""
        discovered = []

        for endpoint in endpoints:
            message = A2AMessage(
                type="request",
                method="discover",
                sender=self.agent_card.id,
                receiver="",
            )

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{endpoint}/a2a",
                        json=message.to_dict(),
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "result" in data.get("payload", {}):
                                card_data = data["payload"]["result"]["agent_card"]
                                card = AgentCard.from_dict(card_data)
                                card.endpoint = endpoint
                                self.register_agent(card)
                                discovered.append(card)

            except Exception as e:
                logger.warning(f"Discovery failed for {endpoint}: {e}")

        return discovered


class A2AServer:
    """
    A2A HTTP Server for receiving messages from other agents.
    """

    def __init__(self, protocol: A2AProtocol, host: str = "0.0.0.0", port: int = 8080):
        self.protocol = protocol
        self.host = host
        self.port = port
        self.app = None
        self.runner = None

    async def start(self):
        """Start the A2A server."""
        from aiohttp import web

        self.app = web.Application()
        self.app.router.add_post("/a2a", self._handle_request)
        self.app.router.add_get("/health", self._handle_health)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

        logger.info(f"A2A server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop the A2A server."""
        if self.runner:
            await self.runner.cleanup()

    async def _handle_request(self, request):
        """Handle incoming A2A request."""
        from aiohttp import web

        try:
            data = await request.json()
            message = A2AMessage.from_dict(data)
            response = await self.protocol.handle_message(message)
            return web.json_response(response.to_dict())

        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _handle_health(self, request):
        """Health check endpoint."""
        from aiohttp import web
        return web.json_response({
            "status": "healthy",
            "agent": self.protocol.agent_card.name,
        })
