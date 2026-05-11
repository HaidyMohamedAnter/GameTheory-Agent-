import numpy as np
from typing import List, Tuple, Dict, Optional

class NormalFormGame:
    """
    Two-player Normal Form (Strategic) Game.
    payoff_A[i,j] = payoff for Player A when A plays i, B plays j.
    payoff_B[i,j] = payoff for Player B when A plays i, B plays j.
    """

    def __init__(self, strategies_A: List[str], strategies_B: List[str],
                 payoff_A: np.ndarray, payoff_B: np.ndarray):
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
                a = self.payoff_A[i, j]
                b = self.payoff_B[i, j]
                row += f"  ({a:.0f},{b:.0f})  "
            print(row)
        print()

    # ── Dominance ──────────────────────────────────────────────────

    def is_strictly_dominated_A(self, strategy_idx: int, remaining_B: List[int]) -> bool:
        for dominator in range(self.n):
            if dominator == strategy_idx:
                continue
            if all(self.payoff_A[dominator, j] > self.payoff_A[strategy_idx, j]
                   for j in remaining_B):
                return True
        return False

    def is_strictly_dominated_B(self, strategy_idx: int, remaining_A: List[int]) -> bool:
        for dominator in range(self.m):
            if dominator == strategy_idx:
                continue
            if all(self.payoff_B[i, dominator] > self.payoff_B[i, strategy_idx]
                   for i in remaining_A):
                return True
        return False

    def iesds(self) -> Tuple[List[str], List[str]]:
        """Iterated Elimination of Strictly Dominated Strategies."""
        rem_A   = list(range(self.n))
        rem_B   = list(range(self.m))
        changed = True
        print("=== IESDS Process ===")
        while changed:
            changed = False
            for i in rem_A[:]:
                if self.is_strictly_dominated_A(i, rem_B):
                    print(f"  Eliminating A: '{self.strategies_A[i]}'")
                    rem_A.remove(i)
                    changed = True
            for j in rem_B[:]:
                if self.is_strictly_dominated_B(j, rem_A):
                    print(f"  Eliminating B: '{self.strategies_B[j]}'")
                    rem_B.remove(j)
                    changed = True
        surviving_A = [self.strategies_A[i] for i in rem_A]
        surviving_B = [self.strategies_B[j] for j in rem_B]
        print(f"  Survivors A: {surviving_A}")
        print(f"  Survivors B: {surviving_B}\n")
        return surviving_A, surviving_B

    # ── Best Response ───────────────────────────────────────────────

    def best_response_A(self, opp_probs: np.ndarray) -> List[int]:
        expected = self.payoff_A @ opp_probs
        max_val  = expected.max()
        return [i for i, v in enumerate(expected) if np.isclose(v, max_val)]

    def best_response_B(self, opp_probs: np.ndarray) -> List[int]:
        expected = self.payoff_B.T @ opp_probs
        max_val  = expected.max()
        return [j for j, v in enumerate(expected) if np.isclose(v, max_val)]

    # ── Pure Nash Equilibrium ───────────────────────────────────────

    def find_pure_nash(self) -> List[Tuple[str, str]]:
        nash = []
        for i in range(self.n):
            for j in range(self.m):
                a_ok = all(self.payoff_A[i, j] >= self.payoff_A[k, j] for k in range(self.n))
                b_ok = all(self.payoff_B[i, j] >= self.payoff_B[i, k] for k in range(self.m))
                if a_ok and b_ok:
                    nash.append((self.strategies_A[i], self.strategies_B[j]))
        return nash

    # ── Mixed Nash Equilibrium (2×2) ────────────────────────────────

    def find_mixed_nash_2x2(self) -> Optional[Tuple[float, float]]:
        if self.n != 2 or self.m != 2:
            return None
        a00, a01 = self.payoff_A[0, 0], self.payoff_A[0, 1]
        a10, a11 = self.payoff_A[1, 0], self.payoff_A[1, 1]
        b00, b01 = self.payoff_B[0, 0], self.payoff_B[0, 1]
        b10, b11 = self.payoff_B[1, 0], self.payoff_B[1, 1]
        denom_q  = (a00 - a01 - a10 + a11)
        denom_p  = (b00 - b01 - b10 + b11)
        if abs(denom_q) < 1e-10 or abs(denom_p) < 1e-10:
            return None
        q_star = (a11 - a01) / denom_q
        p_star = (b11 - b10) / denom_p
        if 0 <= p_star <= 1 and 0 <= q_star <= 1:
            return round(p_star, 4), round(q_star, 4)
        return None


# ── Extensive Form (Sequential Game) ───────────────────────────────

class ExtensiveFormNode:
    def __init__(self, player: str, actions: List[str] = None,
                 payoffs: Dict[str, float] = None):
        self.player   = player
        self.actions  = actions or []
        self.payoffs  = payoffs  or {}
        self.children: Dict[str, 'ExtensiveFormNode'] = {}

    def add_child(self, action: str, node: 'ExtensiveFormNode') -> None:
        self.children[action] = node


def backward_induction(node: ExtensiveFormNode, depth: int = 0) -> Dict[str, float]:
    """Solve sequential game via backward induction."""
    indent = "  " * depth
    if node.player == "terminal":
        print(f"{indent}Terminal payoffs: {node.payoffs}")
        return node.payoffs
    best_payoff = None
    best_action = None
    best_return = {}
    for action, child in node.children.items():
        child_payoff = backward_induction(child, depth + 1)
        player_val   = child_payoff.get(node.player, 0)
        if best_payoff is None or player_val > best_payoff:
            best_payoff = player_val
            best_action = action
            best_return = child_payoff
    print(f"{indent}Player {node.player} chooses '{best_action}' (payoff={best_payoff:.1f})")
    return best_return