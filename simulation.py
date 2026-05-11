import numpy as np
import random
from typing import List, Dict
from agents import Agent, AdaptiveAgent
from game_theory import NormalFormGame

class TrafficEnvironment:
    """
    Smart Traffic Control environment.
    Strategies: 'SHORT' (30s green), 'LONG' (60s green).
    Payoffs represent vehicles cleared per cycle minus congestion penalty.
    """

    STRATEGIES = ['SHORT', 'LONG']
    PAYOFF_A   = np.array([[5, 8], [2, 6]], dtype=float)
    PAYOFF_B   = np.array([[5, 2], [8, 6]], dtype=float)

    def __init__(self, agent_a: Agent, agent_b: Agent, num_rounds: int = 50):
        self.agent_a    = agent_a
        self.agent_b    = agent_b
        self.num_rounds = num_rounds
        self.game       = NormalFormGame(
            self.STRATEGIES, self.STRATEGIES, self.PAYOFF_A, self.PAYOFF_B
        )
        self.round_log: List[Dict] = []

    def get_payoffs(self, action_a: str, action_b: str) -> tuple:
        i = self.STRATEGIES.index(action_a)
        j = self.STRATEGIES.index(action_b)
        return self.PAYOFF_A[i, j], self.PAYOFF_B[i, j]

    def add_noise(self, payoff: float, sigma: float = 0.5) -> float:
        return payoff + np.random.normal(0, sigma)

    def run_episode(self, verbose: bool = False) -> None:
        print(f"\n{'='*50}")
        print(f"SIMULATION: {self.agent_a.agent_id} vs {self.agent_b.agent_id}")
        print(f"{'='*50}")
        for rnd in range(1, self.num_rounds + 1):
            action_a = self.agent_a.choose_action()
            action_b = self.agent_b.choose_action()
            pa, pb   = self.get_payoffs(action_a, action_b)
            pa       = self.add_noise(pa)
            pb       = self.add_noise(pb)
            self.agent_a.update_beliefs(action_b, self.STRATEGIES)
            self.agent_b.update_beliefs(action_a, self.STRATEGIES)
            if isinstance(self.agent_a, AdaptiveAgent):
                self.agent_a.update_q(action_a, pa)
            if isinstance(self.agent_b, AdaptiveAgent):
                self.agent_b.update_q(action_b, pb)
            self.agent_a.record(action_a, pa)
            self.agent_b.record(action_b, pb)
            self.round_log.append({
                'round': rnd, 'action_a': action_a, 'action_b': action_b,
                'payoff_a': round(pa, 2), 'payoff_b': round(pb, 2),
            })
            if verbose and rnd % 10 == 0:
                print(f"  Round {rnd:>2}: A={action_a:<6} B={action_b:<6} "
                      f"Pa={pa:.2f} Pb={pb:.2f}")
        print(f"\nAvg payoff A: {self.agent_a.average_payoff():.3f}")
        print(f"Avg payoff B: {self.agent_b.average_payoff():.3f}")

    def convergence_check(self, window: int = 10) -> str:
        if len(self.agent_a.history) < window:
            return "Insufficient data"
        recent_a = self.agent_a.history[-window:]
        recent_b = self.agent_b.history[-window:]
        if len(set(recent_a)) == 1 and len(set(recent_b)) == 1:
            return f"Converged: A→{recent_a[0]}, B→{recent_b[0]}"
        return "Mixing / not yet converged"