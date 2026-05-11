import numpy as np
import random
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

# ================================================================
#  AGENTS
# ================================================================

class Agent(ABC):
    def __init__(self, agent_id: str, strategies: List[str]):
        self.agent_id        = agent_id
        self.strategies      = strategies
        self.beliefs: Dict[str, float] = {}
        self.history: List[str]        = []
        self.payoff_history: List[float] = []
        self.mixed_strategy  = np.ones(len(strategies)) / len(strategies)

    @abstractmethod
    def choose_action(self) -> str:
        pass

    def update_beliefs(self, opponent_action: str, opponent_strategies: List[str]) -> None:
        if not self.beliefs:
            self.beliefs = {s: 1 / len(opponent_strategies) for s in opponent_strategies}
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
    def __init__(self, agent_id, strategies, fixed_strategy):
        super().__init__(agent_id, strategies)
        self.fixed_strategy = fixed_strategy

    def choose_action(self) -> str:
        return self.fixed_strategy


class MixedStrategyAgent(Agent):
    def __init__(self, agent_id, strategies, probs=None):
        super().__init__(agent_id, strategies)
        if probs:
            self.mixed_strategy = np.array(probs)
            self.mixed_strategy /= self.mixed_strategy.sum()

    def choose_action(self) -> str:
        return np.random.choice(self.strategies, p=self.mixed_strategy)


class BestResponseAgent(Agent):
    def __init__(self, agent_id, strategies, payoff_matrix):
        super().__init__(agent_id, strategies)
        self.payoff_matrix = payoff_matrix

    def choose_action(self) -> str:
        if not self.beliefs:
            return random.choice(self.strategies)
        opp_probs = np.array([self.beliefs.get(s, 0) for s in self.strategies])
        expected  = self.payoff_matrix @ opp_probs
        return self.strategies[int(np.argmax(expected))]


class AdaptiveAgent(Agent):
    def __init__(self, agent_id, strategies, epsilon=0.1, alpha=0.1, gamma=0.9):
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
        self.q_values[action] += self.alpha * (reward - self.q_values[action])


# ================================================================
#  GAME THEORY
# ================================================================

class NormalFormGame:
    def __init__(self, strategies_A, strategies_B, payoff_A, payoff_B):
        self.strategies_A = strategies_A
        self.strategies_B = strategies_B
        self.payoff_A     = np.array(payoff_A, dtype=float)
        self.payoff_B     = np.array(payoff_B, dtype=float)
        self.n = len(strategies_A)
        self.m = len(strategies_B)

    def display_matrix(self) -> None:
        header = f"{'':>12}" + "".join(f"{s:>10}" for s in self.strategies_B)
        print(header)
        print("-" * (12 + 10 * self.m))
        for i, sa in enumerate(self.strategies_A):
            row = f"{sa:>12}"
            for j in range(self.m):
                row += f"  ({self.payoff_A[i,j]:.0f},{self.payoff_B[i,j]:.0f})  "
            print(row)
        print()

    def is_strictly_dominated_A(self, idx, remaining_B):
        for d in range(self.n):
            if d == idx: continue
            if all(self.payoff_A[d, j] > self.payoff_A[idx, j] for j in remaining_B):
                return True
        return False

    def is_strictly_dominated_B(self, idx, remaining_A):
        for d in range(self.m):
            if d == idx: continue
            if all(self.payoff_B[i, d] > self.payoff_B[i, idx] for i in remaining_A):
                return True
        return False

    def iesds(self):
        rem_A, rem_B = list(range(self.n)), list(range(self.m))
        changed = True
        print("=== IESDS Process ===")
        while changed:
            changed = False
            for i in rem_A[:]:
                if self.is_strictly_dominated_A(i, rem_B):
                    print(f"  Eliminating A: '{self.strategies_A[i]}'")
                    rem_A.remove(i); changed = True
            for j in rem_B[:]:
                if self.is_strictly_dominated_B(j, rem_A):
                    print(f"  Eliminating B: '{self.strategies_B[j]}'")
                    rem_B.remove(j); changed = True
        sA = [self.strategies_A[i] for i in rem_A]
        sB = [self.strategies_B[j] for j in rem_B]
        print(f"  Survivors A: {sA}\n  Survivors B: {sB}\n")
        return sA, sB

    def best_response_A(self, opp_probs):
        expected = self.payoff_A @ opp_probs
        max_val  = expected.max()
        return [i for i, v in enumerate(expected) if np.isclose(v, max_val)]

    def best_response_B(self, opp_probs):
        expected = self.payoff_B.T @ opp_probs
        max_val  = expected.max()
        return [j for j, v in enumerate(expected) if np.isclose(v, max_val)]

    def find_pure_nash(self):
        nash = []
        for i in range(self.n):
            for j in range(self.m):
                a_ok = all(self.payoff_A[i,j] >= self.payoff_A[k,j] for k in range(self.n))
                b_ok = all(self.payoff_B[i,j] >= self.payoff_B[i,k] for k in range(self.m))
                if a_ok and b_ok:
                    nash.append((self.strategies_A[i], self.strategies_B[j]))
        return nash

    def find_mixed_nash_2x2(self):
        if self.n != 2 or self.m != 2: return None
        a00,a01 = self.payoff_A[0,0], self.payoff_A[0,1]
        a10,a11 = self.payoff_A[1,0], self.payoff_A[1,1]
        b00,b01 = self.payoff_B[0,0], self.payoff_B[0,1]
        b10,b11 = self.payoff_B[1,0], self.payoff_B[1,1]
        dq = (a00 - a01 - a10 + a11)
        dp = (b00 - b01 - b10 + b11)
        if abs(dq) < 1e-10 or abs(dp) < 1e-10: return None
        q_star = (a11 - a01) / dq
        p_star = (b11 - b10) / dp
        if 0 <= p_star <= 1 and 0 <= q_star <= 1:
            return round(p_star, 4), round(q_star, 4)
        return None


class ExtensiveFormNode:
    def __init__(self, player, actions=None, payoffs=None):
        self.player   = player
        self.actions  = actions or []
        self.payoffs  = payoffs  or {}
        self.children = {}

    def add_child(self, action, node):
        self.children[action] = node


def backward_induction(node, depth=0):
    indent = "  " * depth
    if node.player == "terminal":
        print(f"{indent}Terminal payoffs: {node.payoffs}")
        return node.payoffs
    best_payoff, best_action, best_return = None, None, {}
    for action, child in node.children.items():
        cp  = backward_induction(child, depth + 1)
        val = cp.get(node.player, 0)
        if best_payoff is None or val > best_payoff:
            best_payoff, best_action, best_return = val, action, cp
    print(f"{indent}Player {node.player} chooses '{best_action}' (payoff={best_payoff:.1f})")
    return best_return


# ================================================================
#  SIMULATION
# ================================================================

class TrafficEnvironment:
    STRATEGIES = ['SHORT', 'LONG']
    PAYOFF_A   = np.array([[5, 8], [2, 6]], dtype=float)
    PAYOFF_B   = np.array([[5, 2], [8, 6]], dtype=float)

    def __init__(self, agent_a, agent_b, num_rounds=50):
        self.agent_a    = agent_a
        self.agent_b    = agent_b
        self.num_rounds = num_rounds
        self.round_log  = []

    def get_payoffs(self, action_a, action_b):
        i = self.STRATEGIES.index(action_a)
        j = self.STRATEGIES.index(action_b)
        return self.PAYOFF_A[i, j], self.PAYOFF_B[i, j]

    def add_noise(self, payoff, sigma=0.5):
        return payoff + np.random.normal(0, sigma)

    def run_episode(self, verbose=False):
        print(f"\n{'='*50}")
        print(f"SIMULATION: {self.agent_a.agent_id} vs {self.agent_b.agent_id}")
        print(f"{'='*50}")
        for rnd in range(1, self.num_rounds + 1):
            a = self.agent_a.choose_action()
            b = self.agent_b.choose_action()
            pa, pb = self.get_payoffs(a, b)
            pa, pb = self.add_noise(pa), self.add_noise(pb)
            self.agent_a.update_beliefs(b, self.STRATEGIES)
            self.agent_b.update_beliefs(a, self.STRATEGIES)
            if isinstance(self.agent_a, AdaptiveAgent): self.agent_a.update_q(a, pa)
            if isinstance(self.agent_b, AdaptiveAgent): self.agent_b.update_q(b, pb)
            self.agent_a.record(a, pa)
            self.agent_b.record(b, pb)
            self.round_log.append({'round': rnd, 'action_a': a, 'action_b': b,
                                   'payoff_a': round(pa,2), 'payoff_b': round(pb,2)})
            if verbose and rnd % 10 == 0:
                print(f"  Round {rnd:>3}: A={a:<6} B={b:<6} Pa={pa:.2f} Pb={pb:.2f}")
        print(f"\nAvg payoff A: {self.agent_a.average_payoff():.3f}")
        print(f"Avg payoff B: {self.agent_b.average_payoff():.3f}")

    def convergence_check(self, window=10):
        if len(self.agent_a.history) < window: return "Insufficient data"
        ra = self.agent_a.history[-window:]
        rb = self.agent_b.history[-window:]
        if len(set(ra)) == 1 and len(set(rb)) == 1:
            return f"Converged: A→{ra[0]}, B→{rb[0]}"
        return "Mixing / not yet converged"


#  ANALYSIS & PLOTS


class GameAnalyzer:
    def __init__(self, env):
        self.env     = env
        self.log     = env.round_log
        self.agent_a = env.agent_a
        self.agent_b = env.agent_b

    def summary(self):
        pa = [r['payoff_a'] for r in self.log]
        pb = [r['payoff_b'] for r in self.log]
        def freq(h, s): return h.count(s) / len(h) if h else 0
        stats = {
            'rounds':        len(self.log),
            'avg_payoff_A':  round(np.mean(pa), 3),
            'avg_payoff_B':  round(np.mean(pb), 3),
            'std_payoff_A':  round(np.std(pa),  3),
            'std_payoff_B':  round(np.std(pb),  3),
            'total_welfare': round(np.mean(pa) + np.mean(pb), 3),
            'freq_SHORT_A':  round(freq(self.agent_a.history, 'SHORT'), 3),
            'freq_SHORT_B':  round(freq(self.agent_b.history, 'SHORT'), 3),
        }
        print("\n=== Performance Summary ===")
        for k, v in stats.items():
            print(f"  {k:<22} {v}")
        return stats

    def plot_payoff_evolution(self):
        rounds = [r['round']    for r in self.log]
        pa     = [r['payoff_a'] for r in self.log]
        pb     = [r['payoff_b'] for r in self.log]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(rounds, pa, alpha=0.6, label=self.agent_a.agent_id, color='#185FA5')
        axes[0].plot(rounds, pb, alpha=0.6, label=self.agent_b.agent_id, color='#993C1D')
        axes[0].set_title('Per-round payoffs')
        axes[0].set_xlabel('Round'); axes[0].set_ylabel('Payoff')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(rounds, np.cumsum(pa), label=self.agent_a.agent_id, color='#185FA5')
        axes[1].plot(rounds, np.cumsum(pb), label=self.agent_b.agent_id, color='#993C1D')
        axes[1].set_title('Cumulative payoffs')
        axes[1].set_xlabel('Round'); axes[1].set_ylabel('Cumulative payoff')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('payoff_evolution.png', dpi=150)
        plt.show()
        print("Saved: payoff_evolution.png")

    def plot_strategy_frequency(self, window=10):
        freq_a, freq_b = [], []
        for i in range(len(self.agent_a.history)):
            start = max(0, i - window)
            wa = self.agent_a.history[start:i+1]
            wb = self.agent_b.history[start:i+1]
            freq_a.append(wa.count('SHORT') / len(wa))
            freq_b.append(wb.count('SHORT') / len(wb))
        rounds = range(1, len(freq_a)+1)
        plt.figure(figsize=(10, 4))
        plt.plot(rounds, freq_a, label=f'{self.agent_a.agent_id} SHORT freq', color='#185FA5')
        plt.plot(rounds, freq_b, label=f'{self.agent_b.agent_id} SHORT freq', color='#993C1D')
        plt.axhline(0.5, linestyle='--', color='gray', alpha=0.5, label='50/50 mix')
        plt.title(f'Rolling strategy frequency (window={window})')
        plt.xlabel('Round'); plt.ylabel('Freq(SHORT)')
        plt.ylim(-0.05, 1.05); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('strategy_frequency.png', dpi=150)
        plt.show()
        print("Saved: strategy_frequency.png")

    def plot_belief_evolution(self, beliefs_over_time):
        short_prob = [b.get('SHORT', 0.5) for b in beliefs_over_time]
        rounds     = range(len(short_prob))
        plt.figure(figsize=(8, 3))
        plt.plot(rounds, short_prob, color='#0F6E56')
        plt.axhline(0.5, linestyle='--', color='gray', alpha=0.4)
        plt.fill_between(rounds, short_prob, 0.5, alpha=0.15, color='#0F6E56')
        plt.title('Agent belief: P(opponent plays SHORT)')
        plt.xlabel('Round'); plt.ylabel('Belief probability')
        plt.ylim(0, 1); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('belief_evolution.png', dpi=150)
        plt.show()
        print("Saved: belief_evolution.png")

    def plot_welfare_comparison(self, results):
        labels = list(results.keys())
        values = list(results.values())
        colors = ['#185FA5', '#993C1D', '#0F6E56', '#854F0B']
        plt.figure(figsize=(8, 4))
        bars = plt.bar(labels, values, color=colors[:len(labels)], alpha=0.85, width=0.5)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.1, f'{val:.2f}', ha='center', fontsize=10)
        plt.title('Total social welfare by agent type pairing')
        plt.ylabel('Avg total payoff (A + B)')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('welfare_comparison.png', dpi=150)
        plt.show()
        print("Saved: welfare_comparison.png")


# ================================================================
#  MAIN
# ================================================================

def run_normal_form_analysis():
    print("\n" + "="*55)
    print("SMART TRAFFIC CONTROL – NORMAL FORM GAME ANALYSIS")
    print("="*55)
    strategies = ['SHORT', 'LONG']
    payoff_A   = np.array([[5, 8], [2, 6]], dtype=float)
    payoff_B   = np.array([[5, 2], [8, 6]], dtype=float)
    game       = NormalFormGame(strategies, strategies, payoff_A, payoff_B)
    print("\nPayoff Matrix (Agent A, Agent B):")
    game.display_matrix()
    game.iesds()
    nash = game.find_pure_nash()
    print(f"Pure Nash Equilibria: {nash}")
    mixed = game.find_mixed_nash_2x2()
    if mixed:
        p, q = mixed
        print(f"Mixed Nash: A plays SHORT with p={p}, B plays SHORT with q={q}")
    else:
        print("No interior mixed Nash equilibrium.")
    print("\nBest Responses:")
    for q_val in [0.0, 0.5, 1.0]:
        br    = game.best_response_A(np.array([q_val, 1 - q_val]))
        names = [strategies[i] for i in br]
        print(f"  B plays SHORT w/ prob {q_val:.1f} → A's best response: {names}")


def run_extensive_form():
    print("\n" + "="*55)
    print("EXTENSIVE FORM – SEQUENTIAL INTERSECTION GAME")
    print("="*55)
    t1 = ExtensiveFormNode("terminal", payoffs={"A": 3, "B": 7})
    t2 = ExtensiveFormNode("terminal", payoffs={"A": 2, "B": 2})
    t3 = ExtensiveFormNode("terminal", payoffs={"A": 6, "B": 4})
    t4 = ExtensiveFormNode("terminal", payoffs={"A": 1, "B": 1})
    node_B_yield   = ExtensiveFormNode("B", ["PROCEED", "YIELD"])
    node_B_proceed = ExtensiveFormNode("B", ["YIELD",   "PROCEED"])
    node_B_yield.add_child("PROCEED", t1)
    node_B_yield.add_child("YIELD",   t2)
    node_B_proceed.add_child("YIELD",   t3)
    node_B_proceed.add_child("PROCEED", t4)
    root = ExtensiveFormNode("A", ["YIELD", "PROCEED"])
    root.add_child("YIELD",   node_B_yield)
    root.add_child("PROCEED", node_B_proceed)
    print("\nSolving via backward induction...\n")
    result = backward_induction(root)
    print(f"\nSubgame-perfect equilibrium payoffs: {result}")


def run_agent_simulation():
    strategies = ['SHORT', 'LONG']
    payoff_A   = np.array([[5, 8], [2, 6]], dtype=float)

    agent_a = BestResponseAgent("A_BestResp", strategies, payoff_A)
    agent_b = MixedStrategyAgent("B_Mixed",    strategies, [0.6, 0.4])
    env1    = TrafficEnvironment(agent_a, agent_b, num_rounds=50)
    env1.run_episode(verbose=True)
    print(f"Convergence: {env1.convergence_check()}")

    agent_c = AdaptiveAgent("C_QLearner", strategies, epsilon=0.15)
    agent_d = AdaptiveAgent("D_QLearner", strategies, epsilon=0.15)
    env2    = TrafficEnvironment(agent_c, agent_d, num_rounds=100)
    env2.run_episode(verbose=True)
    print(f"Convergence: {env2.convergence_check()}")


def run_analysis_plots():
    print("\n" + "="*55)
    print("GENERATING PLOTS")
    print("="*55)
    strategies = ['SHORT', 'LONG']
    payoff_A   = np.array([[5, 8], [2, 6]], dtype=float)
    payoff_B   = np.array([[5, 2], [8, 6]], dtype=float)

    agent_a = BestResponseAgent("A_BestResp", strategies, payoff_A)
    agent_b = MixedStrategyAgent("B_Mixed",    strategies, [0.6, 0.4])
    env     = TrafficEnvironment(agent_a, agent_b, num_rounds=50)
    env.run_episode(verbose=False)
    analyzer = GameAnalyzer(env)
    analyzer.summary()
    analyzer.plot_payoff_evolution()
    analyzer.plot_strategy_frequency(window=10)

    beliefs_over_time = []
    temp_agent = BestResponseAgent("temp", strategies, payoff_A)
    for action in agent_b.history:
        temp_agent.update_beliefs(action, strategies)
        beliefs_over_time.append(dict(temp_agent.beliefs))
    analyzer.plot_belief_evolution(beliefs_over_time)

    agent_e = AdaptiveAgent("Q1", strategies, epsilon=0.15)
    agent_f = AdaptiveAgent("Q2", strategies, epsilon=0.15)
    env2    = TrafficEnvironment(agent_e, agent_f, num_rounds=100)
    env2.run_episode(verbose=False)
    a2 = GameAnalyzer(env2)

    agent_g = PureStrategyAgent("Pure_A", strategies, "SHORT")
    agent_h = PureStrategyAgent("Pure_B", strategies, "SHORT")
    env3    = TrafficEnvironment(agent_g, agent_h, num_rounds=50)
    env3.run_episode(verbose=False)
    a3 = GameAnalyzer(env3)

    welfare_results = {
        "BestResp\nvs Mixed":       analyzer.summary()['total_welfare'],
        "Q-learner\nvs Q-learner":  a2.summary()['total_welfare'],
        "Pure SHORT\nvs Pure SHORT": a3.summary()['total_welfare'],
    }
    analyzer.plot_welfare_comparison(welfare_results)
    print("\nAll 4 plots saved in your project folder.")


if __name__ == "__main__":
    run_normal_form_analysis()
    run_extensive_form()
    run_agent_simulation()
    run_analysis_plots()