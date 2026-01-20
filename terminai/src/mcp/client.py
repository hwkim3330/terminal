#!/usr/bin/env python3
"""
TerminaI - MCP (Model Context Protocol) Client

Native MCP support for integrating external tools:
- GitHub
- Postgres/MySQL
- Slack
- Any MCP-compatible server

References:
- https://modelcontextprotocol.io/
"""

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


@dataclass
class MCPServer:
    """MCP server configuration."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # stdio, http, websocket


@dataclass
class MCPResource:
    """MCP resource definition."""
    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


class MCPServerProcess:
    """Manages an MCP server subprocess."""

    def __init__(self, server: MCPServer):
        self.server = server
        self.process: Optional[subprocess.Popen] = None
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []
        self._request_id = 0

    async def start(self) -> bool:
        """Start the MCP server process."""
        try:
            env = {**dict(os.environ), **self.server.env}

            self.process = subprocess.Popen(
                [self.server.command, *self.server.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            # Initialize connection
            await self._initialize()

            # Discover tools
            await self._discover_tools()

            logger.info(f"MCP server '{self.server.name}' started with {len(self.tools)} tools")
            return True

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False

    async def stop(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            self.process = None

    async def _initialize(self):
        """Send initialize request to MCP server."""
        response = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
            },
            "clientInfo": {
                "name": "terminai",
                "version": "1.0.0",
            },
        })

        # Send initialized notification
        await self._send_notification("notifications/initialized", {})

    async def _discover_tools(self):
        """Discover available tools from server."""
        response = await self._send_request("tools/list", {})

        self.tools = []
        for tool_data in response.get("tools", []):
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
                server_name=self.server.name,
            )
            self.tools.append(tool)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        response = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

        # Extract content from response
        content = response.get("content", [])
        if content:
            if content[0].get("type") == "text":
                return content[0].get("text", "")
            return content[0]

        return response

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request to MCP server."""
        if not self.process:
            raise RuntimeError("MCP server not running")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        # Write request
        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line.encode())
        self.process.stdin.flush()

        # Read response
        response_line = self.process.stdout.readline()
        response = json.loads(response_line.decode())

        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")

        return response.get("result", {})

    async def _send_notification(self, method: str, params: Dict[str, Any]):
        """Send JSON-RPC notification to MCP server."""
        if not self.process:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        notification_line = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_line.encode())
        self.process.stdin.flush()


class MCPClient:
    """
    MCP client for managing multiple servers and tools.

    Provides unified interface to all MCP tools across servers.
    """

    # Built-in server configurations
    BUILTIN_SERVERS = {
        "github": MCPServer(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": ""},
        ),
        "filesystem": MCPServer(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/"],
        ),
        "postgres": MCPServer(
            name="postgres",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres"],
            env={"DATABASE_URL": ""},
        ),
        "slack": MCPServer(
            name="slack",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-slack"],
            env={"SLACK_BOT_TOKEN": "", "SLACK_TEAM_ID": ""},
        ),
        "memory": MCPServer(
            name="memory",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
        ),
    }

    def __init__(self):
        self.servers: Dict[str, MCPServerProcess] = {}
        self.tools: Dict[str, MCPTool] = {}

    async def add_server(self, server: MCPServer) -> bool:
        """Add and start an MCP server."""
        process = MCPServerProcess(server)

        if await process.start():
            self.servers[server.name] = process

            # Register tools
            for tool in process.tools:
                self.tools[f"{server.name}/{tool.name}"] = tool

            return True

        return False

    async def add_builtin(self, name: str, **env_vars) -> bool:
        """Add a built-in MCP server."""
        if name not in self.BUILTIN_SERVERS:
            logger.error(f"Unknown built-in server: {name}")
            return False

        server = self.BUILTIN_SERVERS[name]
        server.env.update(env_vars)

        return await self.add_server(server)

    async def remove_server(self, name: str):
        """Remove an MCP server."""
        if name in self.servers:
            await self.servers[name].stop()
            del self.servers[name]

            # Remove tools
            self.tools = {
                k: v for k, v in self.tools.items()
                if not k.startswith(f"{name}/")
            }

    def list_tools(self) -> List[MCPTool]:
        """List all available tools."""
        return list(self.tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get tool by name."""
        # Try full name first
        if name in self.tools:
            return self.tools[name]

        # Try short name
        for full_name, tool in self.tools.items():
            if tool.name == name:
                return tool

        return None

    async def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> Any:
        """
        Call an MCP tool.

        Args:
            name: Tool name (server/tool or just tool)
            arguments: Tool arguments

        Returns:
            Tool result
        """
        arguments = arguments or {}

        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        server = self.servers.get(tool.server_name)
        if not server:
            raise ValueError(f"Server not found: {tool.server_name}")

        return await server.call_tool(tool.name, arguments)

    async def close(self):
        """Close all MCP servers."""
        for server in self.servers.values():
            await server.stop()
        self.servers.clear()
        self.tools.clear()

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get JSON schema for all tools (for LLM function calling)."""
        schemas = []
        for name, tool in self.tools.items():
            schemas.append({
                "name": name,
                "description": tool.description,
                "parameters": tool.input_schema,
            })
        return schemas


# Import os for environment
import os
