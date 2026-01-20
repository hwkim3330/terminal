"""
TerminaI - Sovereign System Operator

A powerful AI agent for system operations with:
- Native PTY support for interactive terminals
- OODA Loop reasoning (Observe-Orient-Decide-Act)
- Gemini 3 API integration (Flash, Pro, Deep Think)
- MCP (Model Context Protocol) for external tools
- A2A (Agent-to-Agent) protocol for orchestration
- Fleet Commander for multi-agent operations

Inspired by Gemini CLI architecture.
"""

__version__ = "1.0.0"
__author__ = "LFM AGI Project"

from .core.agent import TerminaIAgent, AgentState
from .pty.native_pty import NativePTY, PTYSession
from .llm.providers import create_llm_provider, GeminiProvider
from .mcp.client import MCPClient
from .a2a.protocol import A2AProtocol, AgentCard
from .fleet.commander import FleetCommander

__all__ = [
    "TerminaIAgent",
    "AgentState",
    "NativePTY",
    "PTYSession",
    "create_llm_provider",
    "GeminiProvider",
    "MCPClient",
    "A2AProtocol",
    "AgentCard",
    "FleetCommander",
]
