# TerminaI

**Sovereign System Operator** - AI agent with native PTY, OODA reasoning, and fleet orchestration.

Inspired by Gemini CLI architecture. Built for deep system operations.

## Why TerminaI?

| Feature | Claude Cowork | TerminaI |
|---------|--------------|----------|
| **Price** | $100/month | API costs only |
| **Context** | 200K tokens | 2M+ tokens (Gemini 3) |
| **PTY** | Sandboxed | Native PTY |
| **Reasoning** | Basic | OODA Loop |
| **Multi-Agent** | No | A2A + Fleet |
| **MCP** | Limited | Full support |
| **Local LLM** | No | Ollama support |
| **Telemetry** | Yes | None |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TerminaI                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    OODA LOOP                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ OBSERVE  │→ │  ORIENT  │→ │  DECIDE  │→ │   ACT    │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  │       ↑                                          │        │   │
│  │       └──────────────────────────────────────────┘        │   │
│  │                    + VERIFY                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌───────────────────────────┼───────────────────────────────┐  │
│  │                           ▼                               │  │
│  │  ┌──────────┐  ┌──────────────────┐  ┌────────────────┐  │  │
│  │  │Native PTY│  │   Gemini 3 API   │  │  MCP Servers   │  │  │
│  │  │ sudo/ssh │  │ Flash/Pro/Think  │  │ GitHub/Slack/  │  │  │
│  │  │ vim/etc  │  │   2M+ context    │  │ Postgres/etc   │  │  │
│  │  └──────────┘  └──────────────────┘  └────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┼───────────────────────────────┐  │
│  │         FLEET COMMANDER (A2A Protocol)                    │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │  │
│  │  │ Agent 1 │  │ Agent 2 │  │ Agent 3 │  │ Agent N │      │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
cd /mnt/data/lfm_agi/terminai
pip install -e .

# Set API key
export GEMINI_API_KEY='your-key-here'

# Run
python cli.py "Audit /var/log for errors in the last hour"
```

## Usage

### Single Task

```bash
# Basic
terminai "List all running processes"

# With specific model
terminai --model gemini-3-pro "Analyze system performance"

# Using Ollama (local)
terminai --provider ollama --model llama3 "Check disk usage"
```

### Interactive Mode

```bash
terminai --interactive

❯ Check memory usage
❯ Find large files in /home
❯ exit
```

### Fleet Commander

```bash
# Discover and orchestrate multiple agents
terminai --fleet --discover http://agent1:8080 http://agent2:8080
```

### With MCP Servers

```bash
# Enable GitHub integration
export GITHUB_PERSONAL_ACCESS_TOKEN='ghp_...'
terminai --mcp github "Create issue for bug #123"

# Enable multiple servers
terminai --mcp github postgres slack "Deploy and notify team"
```

## Features

### Native PTY

Real terminal handling - not sandboxed:

```python
from terminai import NativePTY

async with NativePTY() as pty:
    # Interactive sudo
    output = await pty.execute_interactive(
        "sudo apt update",
        responses={"password": "mypassword"}
    )

    # SSH tunnel
    await pty.execute("ssh user@server 'uptime'")

    # Vim editing
    await pty.execute_interactive("vim config.yaml")
```

### OODA Loop Reasoning

Self-correcting agent loop:

1. **Observe** - Gather terminal output, file contents, system state
2. **Orient** - Analyze situation, identify problems
3. **Decide** - Choose optimal action with risk assessment
4. **Act** - Execute via PTY/MCP/A2A
5. **Verify** - Validate results, catch errors

```python
from terminai import TerminaIAgent

agent = TerminaIAgent(
    model="gemini-3-flash",
    verification_enabled=True,  # Self-check results
)

result = await agent.run("Deploy the application and verify it's running")
```

### MCP Integration

Connect any MCP server:

```python
from terminai import MCPClient

mcp = MCPClient()

# Built-in servers
await mcp.add_builtin("github", GITHUB_TOKEN="ghp_...")
await mcp.add_builtin("postgres", DATABASE_URL="...")

# Call tools
await mcp.call_tool("github/create_issue", {
    "repo": "myorg/myrepo",
    "title": "Bug report",
})
```

### Fleet Commander (A2A)

Orchestrate multiple agents:

```python
from terminai import FleetCommander

fleet = FleetCommander()

# Discover agents
await fleet.discover(["http://agent1:8080", "http://agent2:8080"])

# Parallel execution
results = await fleet.execute_parallel([
    {"description": "Check server 1"},
    {"description": "Check server 2"},
    {"description": "Check server 3"},
])

# Map-reduce
result = await fleet.map_reduce(
    data=servers,
    map_task="Get metrics from server",
    reduce_task="Aggregate all metrics",
)
```

## Configuration

Environment variables:

```bash
# API Keys
GEMINI_API_KEY=...
GOOGLE_API_KEY=...
OPENAI_API_KEY=...

# MCP Servers
GITHUB_PERSONAL_ACCESS_TOKEN=...
DATABASE_URL=...
SLACK_BOT_TOKEN=...
```

## Models

| Model | Context | Speed | Best For |
|-------|---------|-------|----------|
| gemini-2.0-flash | 1M | Fast | Most tasks |
| gemini-2.0-pro | 2M | Medium | Complex reasoning |
| gemini-3-flash | 2M | Fastest | High throughput |
| gemini-3-pro | 2M | Slow | Deep thinking |

## Sovereign Economics

- **No subscription** - Pay only for API usage
- **Google's free tier** - 15 RPM, 1M TPM free
- **Gemini Pro** - $2.50/1M tokens
- **Self-hosted** - Use Ollama for zero cost
- **No telemetry** - Your data stays local

## vs Claude Cowork

```
                    Claude Cowork          TerminaI
                    ─────────────          ────────
Price               $100/month             ~$2-5/month
Context             200K tokens            2M+ tokens
Terminal            Sandboxed              Native PTY
Interactive         Limited                Full (sudo/ssh/vim)
Multi-agent         No                     A2A + Fleet
MCP Support         Limited                Full
Local LLMs          No                     Ollama/vLLM
Data Privacy        Cloud                  Local execution
```

## License

Apache-2.0

## References

- [Google Gemini](https://deepmind.google/technologies/gemini/)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [A2A Protocol](https://github.com/google/a2a)
