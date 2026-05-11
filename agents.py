import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class Agent(ABC):
    """Abstract base class for all intelligent agents."""

    def __init__(self, agent_id: str, strategies: List[str]):
        self.agent_id   = agent_id
        self.strategies = strategies
        self.beliefs: Dict[str, float] = {}
        self.history: List[str]        = []
        self.payoff_history: List[float] = []
        self.mixed_strategy: np.ndarray = (
            np.ones(len(strategies)) / len(strategies)
        )

    @abstractmethod
    def choose_action(self) -> str:
        pass

    def update_beliefs(self, opponent_action: str,
                       opponent_strategies: List[str]) -> None:
        """Bayesian belief update about opponent behavior."""
        if not self.beliefs:
            self.beliefs = {s: 1 / len(opponent_strategies)
                            for s in opponent_strategies}
        alpha = 0.3
        for s in opponent_strategies:
            if s == opponent_action:
                self.beliefs[s] = (1 - alpha) * self.beliefs[s] + alpha * 1.0
            else:
                self.beliefs[s] = (1 - alpha) * self.beliefs[s] + alpha * 0.0
        total = sum(self.beliefs.values())
        self.beliefs = {s: v / total for s, v in self.beliefs.items()}

    def record(self, action: str, payoff: float) -> None:
        self.history.append(action)
        self.payoff_history.append(payoff)

    def average_payoff(self) -> float:
        return np.mean(self.payoff_history) if self.payoff_history else 0.0


class PureStrategyAgent(Agent):
    """Agent that always plays the same pure strategy."""

    def __init__(self, agent_id: str, strategies: List[str], fixed_strategy: str):
        super().__init__(agent_id, strategies)
        self.fixed_strategy = fixed_strategy

    def choose_action(self) -> str:
        return self.fixed_strategy


class MixedStrategyAgent(Agent):
    """Agent that randomises over strategies using a probability vector."""

    def __init__(self, agent_id: str, strategies: List[str], probs: List[float] = None):
        super().__init__(agent_id, strategies)
        if probs:
            self.mixed_strategy = np.array(probs)
            self.mixed_strategy /= self.mixed_strategy.sum()

    def choose_action(self) -> str:
        return np.random.choice(self.strategies, p=self.mixed_strategy)


class BestResponseAgent(Agent):
    """Agent that computes and plays the best response to its beliefs."""

    def __init__(self, agent_id: str, strategies: List[str], payoff_matrix: np.ndarray):
        super().__init__(agent_id, strategies)
        self.payoff_matrix = payoff_matrix

    def choose_action(self) -> str:
        if not self.beliefs:
            return random.choice(self.strategies)
        opp_probs = np.array([self.beliefs.get(s, 0) for s in self.strategies])
        expected  = self.payoff_matrix @ opp_probs
        best_idx  = int(np.argmax(expected))
        return self.strategies[best_idx]


class AdaptiveAgent(Agent):
    """Reinforcement-learning agent that adapts via Q-learning."""

    def __init__(self, agent_id: str, strategies: List[str],
                 epsilon: float = 0.1, alpha: float = 0.1, gamma: float = 0.9):
        super().__init__(agent_id, strategies)
        self.epsilon  = epsilon
        self.alpha    = alpha
        self.gamma    = gamma
        self.q_values = {s: 0.0 for s in strategies}

    def choose_action(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.strategies)
        return max(self.q_values, key=self.q_values.get)

    def update_q(self, action: str, reward: float) -> None:
        """Q(a) ← Q(a) + α[r - Q(a)]"""
        self.q_values[action] += self.alpha * (reward - self.q_values[action])