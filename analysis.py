
import matplotlib
matplotlib.use('TkAgg')   # use this on Windows/Linux
# matplotlib.use('MacOSX') # use this on Mac

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from agents import Agent
from simulation import TrafficEnvironment


class GameAnalyzer:
    def __init__(self, env: TrafficEnvironment):
        self.env     = env
        self.log     = env.round_log
        self.agent_a = env.agent_a
        self.agent_b = env.agent_b

    def summary(self) -> Dict:
        payoffs_a = [r['payoff_a'] for r in self.log]
        payoffs_b = [r['payoff_b'] for r in self.log]

        def freq(hist, strat):
            return hist.count(strat) / len(hist) if hist else 0

        stats = {
            'rounds':        len(self.log),
            'avg_payoff_A':  round(np.mean(payoffs_a), 3),
            'avg_payoff_B':  round(np.mean(payoffs_b), 3),
            'std_payoff_A':  round(np.std(payoffs_a),  3),
            'std_payoff_B':  round(np.std(payoffs_b),  3),
            'total_welfare': round(np.mean(payoffs_a) + np.mean(payoffs_b), 3),
            'freq_SHORT_A':  round(freq(self.agent_a.history, 'SHORT'), 3),
            'freq_SHORT_B':  round(freq(self.agent_b.history, 'SHORT'), 3),
        }
        print("\n=== Performance Summary ===")
        for k, v in stats.items():
            print(f"  {k:<22} {v}")
        return stats

    def plot_payoff_evolution(self) -> None:
        rounds    = [r['round']    for r in self.log]
        payoffs_a = [r['payoff_a'] for r in self.log]
        payoffs_b = [r['payoff_b'] for r in self.log]
        cum_a     = np.cumsum(payoffs_a)
        cum_b     = np.cumsum(payoffs_b)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(rounds, payoffs_a, alpha=0.6, label=self.agent_a.agent_id, color='#185FA5')
        axes[0].plot(rounds, payoffs_b, alpha=0.6, label=self.agent_b.agent_id, color='#993C1D')
        axes[0].set_title('Per-round payoffs')
        axes[0].set_xlabel('Round'); axes[0].set_ylabel('Payoff')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(rounds, cum_a, label=self.agent_a.agent_id, color='#185FA5')
        axes[1].plot(rounds, cum_b, label=self.agent_b.agent_id, color='#993C1D')
        axes[1].set_title('Cumulative payoffs')
        axes[1].set_xlabel('Round'); axes[1].set_ylabel('Cumulative payoff')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('payoff_evolution.png', dpi=150)
        plt.show()

    def plot_strategy_frequency(self, window: int = 10) -> None:
        rounds = range(1, len(self.agent_a.history) + 1)
        freq_a, freq_b = [], []
        for i in range(len(self.agent_a.history)):
            start    = max(0, i - window)
            window_a = self.agent_a.history[start:i+1]
            window_b = self.agent_b.history[start:i+1]
            freq_a.append(window_a.count('SHORT') / len(window_a))
            freq_b.append(window_b.count('SHORT') / len(window_b))

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

    def plot_belief_evolution(self, beliefs_over_time: List[Dict]) -> None:
        rounds     = range(len(beliefs_over_time))
        short_prob = [b.get('SHORT', 0.5) for b in beliefs_over_time]
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

    def plot_welfare_comparison(self, results: Dict[str, float]) -> None:
        labels = list(results.keys())
        values = list(results.values())
        colors = ['#185FA5', '#993C1D', '#0F6E56', '#854F0B']
        plt.figure(figsize=(8, 4))
        bars = plt.bar(labels, values, color=colors[:len(labels)], alpha=0.85, width=0.5)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val:.2f}', ha='center', fontsize=10)
        plt.title('Total social welfare by agent type pairing')
        plt.ylabel('Avg total payoff (A + B)')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('welfare_comparison.png', dpi=150)
        plt.show()