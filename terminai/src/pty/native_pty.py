#!/usr/bin/env python3
"""
TerminaI - Native PTY Handler

Provides real terminal handling for interactive sessions.
Inspired by Gemini CLI's node-pty architecture.

Handles:
- Interactive sudo prompts
- SSH tunnels
- vim/nano sessions
- Any interactive terminal application
"""

import asyncio
import os
import pty
import select
import signal
import subprocess
import sys
import termios
import tty
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import fcntl
import struct
import logging
import re
import time

logger = logging.getLogger(__name__)


@dataclass
class PTYConfig:
    """PTY configuration."""
    rows: int = 24
    cols: int = 80
    shell: str = "/bin/bash"
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    timeout: float = 30.0


class NativePTY:
    """
    Native PTY implementation for real terminal handling.

    Unlike simple subprocess, this properly handles:
    - Password prompts (sudo, ssh)
    - Interactive editors (vim, nano)
    - Terminal escape sequences
    - Job control (Ctrl+C, Ctrl+Z)
    - Window resize
    """

    # Common prompts to detect
    PROMPTS = {
        "password": [
            r"\[sudo\] password for",
            r"Password:",
            r"password:",
            r"Enter passphrase",
        ],
        "confirmation": [
            r"\[y/N\]",
            r"\[Y/n\]",
            r"Are you sure",
            r"Do you want to continue",
        ],
        "shell": [
            r"\$\s*$",
            r"#\s*$",
            r">\s*$",
        ],
    }

    def __init__(self, config: Optional[PTYConfig] = None):
        self.config = config or PTYConfig()

        # PTY state
        self.master_fd: Optional[int] = None
        self.slave_fd: Optional[int] = None
        self.pid: Optional[int] = None
        self.is_running: bool = False

        # Output buffer
        self.output_buffer: str = ""
        self.last_exit_code: int = 0

        # Callbacks
        self.on_output: Optional[Callable[[str], None]] = None
        self.on_prompt: Optional[Callable[[str, str], Optional[str]]] = None

    def start(self) -> bool:
        """Start PTY session."""
        if self.is_running:
            return True

        try:
            # Create PTY pair
            self.master_fd, self.slave_fd = pty.openpty()

            # Set window size
            self._set_window_size(self.config.rows, self.config.cols)

            # Fork process
            self.pid = os.fork()

            if self.pid == 0:
                # Child process
                self._child_process()
            else:
                # Parent process
                os.close(self.slave_fd)
                self.slave_fd = None
                self.is_running = True
                logger.info(f"PTY started with PID {self.pid}")

            return True

        except Exception as e:
            logger.error(f"Failed to start PTY: {e}")
            return False

    def _child_process(self):
        """Setup child process."""
        # Create new session
        os.setsid()

        # Set controlling terminal
        os.close(self.master_fd)

        # Duplicate slave to stdin/stdout/stderr
        os.dup2(self.slave_fd, 0)
        os.dup2(self.slave_fd, 1)
        os.dup2(self.slave_fd, 2)

        if self.slave_fd > 2:
            os.close(self.slave_fd)

        # Set environment
        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        if self.config.env:
            env.update(self.config.env)

        # Change directory
        if self.config.cwd:
            os.chdir(self.config.cwd)

        # Execute shell
        os.execvpe(self.config.shell, [self.config.shell], env)

    def stop(self):
        """Stop PTY session."""
        if not self.is_running:
            return

        try:
            # Send SIGTERM
            if self.pid:
                os.kill(self.pid, signal.SIGTERM)
                os.waitpid(self.pid, 0)

            # Close master
            if self.master_fd:
                os.close(self.master_fd)

            self.is_running = False
            self.master_fd = None
            self.pid = None
            logger.info("PTY stopped")

        except Exception as e:
            logger.error(f"Error stopping PTY: {e}")

    def _set_window_size(self, rows: int, cols: int):
        """Set PTY window size."""
        if self.master_fd is None:
            return

        try:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
        except Exception as e:
            logger.warning(f"Failed to set window size: {e}")

    def resize(self, rows: int, cols: int):
        """Resize PTY window."""
        self.config.rows = rows
        self.config.cols = cols
        self._set_window_size(rows, cols)

    def write(self, data: str) -> bool:
        """Write data to PTY."""
        if not self.is_running or self.master_fd is None:
            return False

        try:
            os.write(self.master_fd, data.encode())
            return True
        except Exception as e:
            logger.error(f"Write error: {e}")
            return False

    def write_line(self, data: str) -> bool:
        """Write line to PTY (adds newline)."""
        return self.write(data + "\n")

    def read_output(self, timeout: float = 0.1) -> str:
        """Read available output from PTY."""
        if not self.is_running or self.master_fd is None:
            return ""

        output = ""
        try:
            while True:
                r, _, _ = select.select([self.master_fd], [], [], timeout)
                if not r:
                    break

                data = os.read(self.master_fd, 4096)
                if not data:
                    break

                chunk = data.decode("utf-8", errors="replace")
                output += chunk
                self.output_buffer += chunk

                if self.on_output:
                    self.on_output(chunk)

        except Exception as e:
            logger.error(f"Read error: {e}")

        return output

    async def execute(self, command: str, timeout: float = 30.0) -> str:
        """
        Execute command and wait for completion.

        Args:
            command: Command to execute
            timeout: Maximum wait time

        Returns:
            Command output
        """
        if not self.is_running:
            self.start()

        # Clear buffer
        self.output_buffer = ""

        # Write command
        self.write_line(command)

        # Wait for output
        output = await self._wait_for_completion(timeout)

        return output

    async def execute_interactive(
        self,
        command: str,
        timeout: float = 60.0,
        responses: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Execute interactive command with automatic responses.

        Args:
            command: Command to execute
            timeout: Maximum wait time
            responses: Dict mapping prompt patterns to responses

        Returns:
            Command output
        """
        if not self.is_running:
            self.start()

        responses = responses or {}
        self.output_buffer = ""

        # Write command
        self.write_line(command)

        # Monitor output and respond to prompts
        start_time = time.time()
        last_output_time = start_time

        while time.time() - start_time < timeout:
            output = self.read_output(timeout=0.5)

            if output:
                last_output_time = time.time()

                # Check for prompts
                response = self._check_prompts(output, responses)
                if response:
                    await asyncio.sleep(0.1)
                    self.write_line(response)
                    continue

            # Check if command completed
            if self._is_shell_prompt(self.output_buffer):
                if time.time() - last_output_time > 0.5:
                    break

            await asyncio.sleep(0.1)

        return self.output_buffer

    async def _wait_for_completion(self, timeout: float) -> str:
        """Wait for command completion."""
        start_time = time.time()
        last_output_time = start_time

        while time.time() - start_time < timeout:
            output = self.read_output(timeout=0.5)

            if output:
                last_output_time = time.time()

            # Check if shell prompt appeared
            if self._is_shell_prompt(self.output_buffer):
                if time.time() - last_output_time > 0.3:
                    break

            await asyncio.sleep(0.1)

        return self.output_buffer

    def _check_prompts(self, output: str, responses: Dict[str, str]) -> Optional[str]:
        """Check for prompts in output."""
        # Check password prompts
        for pattern in self.PROMPTS["password"]:
            if re.search(pattern, output, re.IGNORECASE):
                if "password" in responses:
                    return responses["password"]
                if self.on_prompt:
                    return self.on_prompt("password", output)

        # Check confirmation prompts
        for pattern in self.PROMPTS["confirmation"]:
            if re.search(pattern, output, re.IGNORECASE):
                if "confirm" in responses:
                    return responses["confirm"]
                if self.on_prompt:
                    return self.on_prompt("confirm", output)

        # Check custom responses
        for pattern, response in responses.items():
            if pattern not in ["password", "confirm"]:
                if re.search(pattern, output, re.IGNORECASE):
                    return response

        return None

    def _is_shell_prompt(self, output: str) -> bool:
        """Check if output ends with shell prompt."""
        # Clean ANSI codes
        clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', output)
        clean = clean.strip()

        for pattern in self.PROMPTS["shell"]:
            if re.search(pattern, clean):
                return True

        return False

    def send_interrupt(self):
        """Send Ctrl+C (SIGINT)."""
        if self.pid:
            os.kill(self.pid, signal.SIGINT)

    def send_eof(self):
        """Send Ctrl+D (EOF)."""
        self.write("\x04")

    def send_suspend(self):
        """Send Ctrl+Z (SIGTSTP)."""
        if self.pid:
            os.kill(self.pid, signal.SIGTSTP)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class PTYSession:
    """
    High-level PTY session manager.

    Manages multiple PTY instances and provides
    convenient methods for common operations.
    """

    def __init__(self):
        self.sessions: Dict[str, NativePTY] = {}
        self.default_session: Optional[str] = None

    def create(self, name: str = "default", config: Optional[PTYConfig] = None) -> NativePTY:
        """Create new PTY session."""
        if name in self.sessions:
            self.sessions[name].stop()

        pty_instance = NativePTY(config)
        pty_instance.start()

        self.sessions[name] = pty_instance
        if self.default_session is None:
            self.default_session = name

        return pty_instance

    def get(self, name: Optional[str] = None) -> Optional[NativePTY]:
        """Get PTY session by name."""
        name = name or self.default_session
        if name is None:
            return None
        return self.sessions.get(name)

    def close(self, name: Optional[str] = None):
        """Close PTY session."""
        name = name or self.default_session
        if name and name in self.sessions:
            self.sessions[name].stop()
            del self.sessions[name]

            if self.default_session == name:
                self.default_session = next(iter(self.sessions), None)

    def close_all(self):
        """Close all PTY sessions."""
        for pty_instance in self.sessions.values():
            pty_instance.stop()
        self.sessions.clear()
        self.default_session = None

    async def run(
        self,
        command: str,
        session: Optional[str] = None,
        interactive: bool = False,
        **kwargs,
    ) -> str:
        """Run command in PTY session."""
        pty_instance = self.get(session)
        if pty_instance is None:
            pty_instance = self.create(session or "default")

        if interactive:
            return await pty_instance.execute_interactive(command, **kwargs)
        else:
            return await pty_instance.execute(command, **kwargs)
