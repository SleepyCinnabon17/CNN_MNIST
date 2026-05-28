"""Optimizers and learning-rate schedules."""

from __future__ import annotations

import math

import numpy as np


def _decay_applies(key: str, value: np.ndarray) -> bool:
    return key.split(".")[-1] in {"W", "gamma"} and value.ndim >= 1


class SGDMomentum:
    """Stochastic gradient descent with momentum and optional L2 weight decay."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """Update parameters in place."""
        for key, param in params.items():
            grad = grads[key].copy()
            if self.weight_decay and _decay_applies(key, param):
                grad = grad + self.weight_decay * param
            velocity = self.velocity.get(key)
            if velocity is None:
                velocity = np.zeros_like(param)
            velocity = self.momentum * velocity - self.lr * grad
            param += velocity
            self.velocity[key] = velocity


class Adam:
    """Adam optimizer with optional L2 weight decay."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """Update parameters in place."""
        self.t += 1
        for key, param in params.items():
            grad = grads[key].copy()
            if self.weight_decay and _decay_applies(key, param):
                grad = grad + self.weight_decay * param
            m = self.m.get(key)
            v = self.v.get(key)
            if m is None or v is None:
                m = np.zeros_like(param)
                v = np.zeros_like(param)
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.m[key] = m
            self.v[key] = v


def step_decay_lr(base_lr: float, epoch: int, decay_factor: float, decay_epochs: int) -> float:
    """Return a step-decayed learning rate."""
    return base_lr * (decay_factor ** (epoch // decay_epochs))


def cosine_lr(base_lr: float, epoch: int, total_epochs: int) -> float:
    """Return a cosine-annealed learning rate."""
    if total_epochs <= 0:
        return 0.0
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
