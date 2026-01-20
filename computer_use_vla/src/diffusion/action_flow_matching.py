# SPDX-License-Identifier: Apache-2.0
# Computer Use VLA - Action Flow Matching
# Inspired by Alpamayo's diffusion-based trajectory decoder

"""
Flow Matching for Computer Action Generation.

Uses flow matching to generate smooth mouse trajectories
and natural action sequences.

References:
- Flow Matching for Generative Modeling (https://arxiv.org/pdf/2210.02747)
- NVIDIA Alpamayo Flow Matching implementation
"""

from typing import Callable, Literal, Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np


StepFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching."""
    x_dims: Tuple[int, ...] = (16, 2)  # trajectory_length x (x, y)
    num_inference_steps: int = 10
    integration_method: str = "euler"
    sigma_min: float = 0.001
    sigma_max: float = 1.0


class BaseFlowMatching(nn.Module):
    """Base class for flow matching models."""

    def __init__(
        self,
        x_dims: Tuple[int, ...],
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.x_dims = x_dims
        self.num_inference_steps = num_inference_steps

    def get_timestep_embedding(
        self,
        timesteps: torch.Tensor,
        embedding_dim: int = 128,
    ) -> torch.Tensor:
        """
        Sinusoidal timestep embeddings.

        Args:
            timesteps: [B] or [B, 1]
            embedding_dim: Dimension of embeddings

        Returns:
            embeddings: [B, embedding_dim]
        """
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)

        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn: StepFn,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
    ) -> torch.Tensor:
        """Sample from the flow matching model."""
        raise NotImplementedError


class ActionFlowMatching(BaseFlowMatching):
    """
    Flow matching model for computer action generation.

    Generates smooth mouse trajectories and action sequences
    using flow-based generative modeling.
    """

    def __init__(
        self,
        trajectory_length: int = 16,
        action_dim: int = 2,  # (x, y) or more for full action
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_inference_steps: int = 10,
        integration_method: Literal["euler", "heun"] = "euler",
    ):
        x_dims = (trajectory_length, action_dim)
        super().__init__(x_dims, num_inference_steps)

        self.trajectory_length = trajectory_length
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.integration_method = integration_method

        # Velocity network
        self.velocity_net = self._build_velocity_network(num_layers)

    def _build_velocity_network(self, num_layers: int) -> nn.Module:
        """Build velocity prediction network."""
        input_dim = self.trajectory_length * self.action_dim + 128  # + time embedding

        layers = []
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(self.hidden_dim))

        layers.append(nn.Linear(self.hidden_dim, self.trajectory_length * self.action_dim))

        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity at given state and time.

        Args:
            x: Current state [B, trajectory_length, action_dim]
            t: Time [B, 1] in [0, 1]
            condition: Optional conditioning [B, hidden_dim]

        Returns:
            velocity: [B, trajectory_length, action_dim]
        """
        batch_size = x.shape[0]

        # Flatten trajectory
        x_flat = x.view(batch_size, -1)

        # Get time embedding
        t_emb = self.get_timestep_embedding(t.squeeze(-1), 128)

        # Concatenate
        features = torch.cat([x_flat, t_emb], dim=-1)

        # Predict velocity
        velocity = self.velocity_net(features)

        # Reshape
        velocity = velocity.view(batch_size, self.trajectory_length, self.action_dim)

        return velocity

    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            x_0: Noise samples [B, trajectory_length, action_dim]
            x_1: Target trajectories [B, trajectory_length, action_dim]
            condition: Optional conditioning

        Returns:
            loss: Scalar loss
        """
        batch_size = x_0.shape[0]

        # Sample time uniformly
        t = torch.rand(batch_size, 1, device=x_0.device)

        # Interpolate
        x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1

        # True velocity
        v_true = x_1 - x_0

        # Predicted velocity
        v_pred = self(x_t, t, condition)

        # MSE loss
        loss = nn.functional.mse_loss(v_pred, v_true)

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn: Optional[StepFn] = None,
        device: torch.device = torch.device("cpu"),
        condition: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
    ) -> torch.Tensor:
        """
        Sample trajectories using flow matching.

        Args:
            batch_size: Number of samples
            step_fn: Optional custom step function
            device: Device to use
            condition: Optional conditioning
            return_all_steps: Return intermediate steps

        Returns:
            samples: [B, trajectory_length, action_dim] or list of steps
        """
        # Start from noise
        x = torch.randn(batch_size, *self.x_dims, device=device)

        # Time steps
        time_steps = torch.linspace(0, 1, self.num_inference_steps + 1, device=device)

        if return_all_steps:
            all_steps = [x.clone()]

        # Integration
        for i in range(self.num_inference_steps):
            t = time_steps[i].expand(batch_size, 1)
            dt = time_steps[i + 1] - time_steps[i]

            if step_fn is not None:
                v = step_fn(x, t)
            else:
                v = self(x, t, condition)

            if self.integration_method == "euler":
                x = x + dt * v
            elif self.integration_method == "heun":
                # Heun's method (predictor-corrector)
                x_pred = x + dt * v
                t_next = time_steps[i + 1].expand(batch_size, 1)
                v_next = self(x_pred, t_next, condition) if step_fn is None else step_fn(x_pred, t_next)
                x = x + dt * 0.5 * (v + v_next)

            if return_all_steps:
                all_steps.append(x.clone())

        if return_all_steps:
            return torch.stack(all_steps, dim=1)

        return x


class MouseTrajectoryGenerator(nn.Module):
    """
    Generates smooth mouse trajectories for computer control.

    Uses flow matching to create natural-looking cursor movements
    from start to end position.
    """

    def __init__(
        self,
        trajectory_length: int = 16,
        hidden_dim: int = 256,
        num_inference_steps: int = 10,
    ):
        super().__init__()

        self.trajectory_length = trajectory_length

        # Flow matching for trajectory
        self.flow_matcher = ActionFlowMatching(
            trajectory_length=trajectory_length,
            action_dim=2,  # (x, y)
            hidden_dim=hidden_dim,
            num_inference_steps=num_inference_steps,
        )

        # Conditioning encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # start_x, start_y, end_x, end_y
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def generate_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Generate smooth trajectory from start to end.

        Args:
            start_pos: (x, y) starting position (normalized 0-1)
            end_pos: (x, y) ending position (normalized 0-1)
            device: Device to use

        Returns:
            trajectory: [trajectory_length, 2] positions
        """
        # Create condition
        condition = torch.tensor([
            start_pos[0], start_pos[1],
            end_pos[0], end_pos[1]
        ], device=device).unsqueeze(0)

        cond_emb = self.condition_encoder(condition)

        # Sample trajectory
        trajectory = self.flow_matcher.sample(
            batch_size=1,
            device=device,
            condition=cond_emb,
        ).squeeze(0)

        # Clamp to valid range
        trajectory = torch.clamp(trajectory, 0, 1)

        # Ensure start and end points match
        trajectory[0] = torch.tensor(start_pos, device=device)
        trajectory[-1] = torch.tensor(end_pos, device=device)

        return trajectory

    def generate_bezier_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        curvature: float = 0.2,
    ) -> torch.Tensor:
        """
        Generate trajectory using cubic Bezier curve.

        Faster than flow matching, useful for inference.

        Args:
            start_pos: Starting position
            end_pos: Ending position
            curvature: Control point offset (0 = straight line)

        Returns:
            trajectory: [trajectory_length, 2]
        """
        t = torch.linspace(0, 1, self.trajectory_length)

        # Control points
        p0 = torch.tensor(start_pos)
        p3 = torch.tensor(end_pos)

        # Add perpendicular offset for natural curve
        direction = p3 - p0
        perpendicular = torch.tensor([-direction[1], direction[0]])
        perpendicular = perpendicular / (perpendicular.norm() + 1e-8)

        offset = curvature * (torch.rand(1) - 0.5) * 2
        p1 = p0 + direction * 0.33 + perpendicular * offset
        p2 = p0 + direction * 0.66 + perpendicular * offset

        # Cubic Bezier
        trajectory = (
            (1 - t).unsqueeze(-1) ** 3 * p0 +
            3 * (1 - t).unsqueeze(-1) ** 2 * t.unsqueeze(-1) * p1 +
            3 * (1 - t).unsqueeze(-1) * t.unsqueeze(-1) ** 2 * p2 +
            t.unsqueeze(-1) ** 3 * p3
        )

        return trajectory.clamp(0, 1)


class ActionSequenceGenerator(nn.Module):
    """
    Generates sequences of computer actions using flow matching.

    Learns to produce multi-step action sequences for complex tasks.
    """

    def __init__(
        self,
        max_sequence_length: int = 10,
        action_dim: int = 10,
        hidden_dim: int = 512,
        num_inference_steps: int = 10,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.action_dim = action_dim

        # Flow matcher for action sequences
        self.flow_matcher = ActionFlowMatching(
            trajectory_length=max_sequence_length,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_inference_steps=num_inference_steps,
        )

        # Task embedding
        self.task_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            ),
            num_layers=2,
        )

    def forward(
        self,
        task_embedding: torch.Tensor,
        screen_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate action sequence given task and screen.

        Args:
            task_embedding: [B, seq_len, hidden_dim]
            screen_features: [B, hidden_dim]

        Returns:
            action_sequence: [B, max_sequence_length, action_dim]
        """
        # Encode task
        task_encoded = self.task_encoder(task_embedding)
        task_pooled = task_encoded.mean(dim=1)

        # Combine with screen features
        condition = task_pooled + screen_features

        # Generate action sequence
        action_sequence = self.flow_matcher.sample(
            batch_size=task_embedding.shape[0],
            device=task_embedding.device,
            condition=condition.unsqueeze(1),
        )

        return action_sequence
