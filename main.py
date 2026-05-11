import numpy as np
from agents import PureStrategyAgent, MixedStrategyAgent, BestResponseAgent, AdaptiveAgent
from game_theory import NormalFormGame, ExtensiveFormNode, backward_induction
from simulation import TrafficEnvironment

def run_normal_form_analysis() -> None:
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


def run_extensive_form() -> None:
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


def run_agent_simulation() -> None:
    strategies = ['SHORT', 'LONG']
    payoff_A   = np.array([[5, 8], [2, 6]], dtype=float)

    # Scenario 1: Best Response vs Mixed Strategy
    agent_a = BestResponseAgent("A_BestResp", strategies, payoff_A)
    agent_b = MixedStrategyAgent("B_Mixed",    strategies, [0.6, 0.4])
    env1    = TrafficEnvironment(agent_a, agent_b, num_rounds=50)
    env1.run_episode(verbose=True)
    print(f"Convergence: {env1.convergence_check()}")

    # Scenario 2: Two Q-learning agents
    agent_c = AdaptiveAgent("C_QLearner", strategies, epsilon=0.15)
    agent_d = AdaptiveAgent("D_QLearner", strategies, epsilon=0.15)
    env2    = TrafficEnvironment(agent_c, agent_d, num_rounds=100)
    env2.run_episode(verbose=True)
    print(f"Convergence: {env2.convergence_check()}")


if __name__ == "__main__":
    run_normal_form_analysis()
    run_extensive_form()
    run_agent_simulation()

    from analysis import GameAnalyzer

def run_analysis_plots() -> None:

    strategies = ['SHORT', 'LONG']
    payoff_A   = np.array([[5, 8], [2, 6]], dtype=float)
    payoff_B   = np.array([[5, 2], [8, 6]], dtype=float)

    # --- Run simulation to get data ---
    agent_a = BestResponseAgent("A_BestResp", strategies, payoff_A)
    agent_b = MixedStrategyAgent("B_Mixed",    strategies, [0.6, 0.4])
    env     = TrafficEnvironment(agent_a, agent_b, num_rounds=50)
    env.run_episode(verbose=False)

    analyzer = GameAnalyzer(env)

    # --- Plot 1: Payoff Evolution ---
    analyzer.plot_payoff_evolution()

    # --- Plot 2: Strategy Frequency ---
    analyzer.plot_strategy_frequency(window=10)

    # --- Plot 3: Belief Evolution ---
    beliefs_over_time = []
    for action in agent_a.history:
        agent_a.update_beliefs(action, strategies)
        beliefs_over_time.append(dict(agent_a.beliefs))
    analyzer.plot_belief_evolution(beliefs_over_time)

    # --- Plot 4: Welfare Comparison (run all scenarios) ---
    agent_c = BestResponseAgent("BestResp", strategies, payoff_A)
    agent_d = MixedStrategyAgent("Mixed",    strategies, [0.6, 0.4])
    env2    = TrafficEnvironment(agent_c, agent_d, num_rounds=50)
    env2.run_episode()
    a2 = GameAnalyzer(env2)

    agent_e = AdaptiveAgent("Q1", strategies, epsilon=0.15)
    agent_f = AdaptiveAgent("Q2", strategies, epsilon=0.15)
    env3    = TrafficEnvironment(agent_e, agent_f, num_rounds=100)
    env3.run_episode()
    a3 = GameAnalyzer(env3)

    agent_g = PureStrategyAgent("Pure_SHORT_A", strategies, "SHORT")
    agent_h = PureStrategyAgent("Pure_SHORT_B", strategies, "SHORT")
    env4    = TrafficEnvironment(agent_g, agent_h, num_rounds=50)
    env4.run_episode()
    a4 = GameAnalyzer(env4)

    welfare_results = {
        "BestResp\nvs Mixed":   a2.summary()['total_welfare'],
        "Q-learner\nvs Q-learner": a3.summary()['total_welfare'],
        "Pure SHORT\nvs Pure SHORT": a4.summary()['total_welfare'],
    }
    analyzer.plot_welfare_comparison(welfare_results)

    print("\nAll plots saved as PNG files in your project folder.")


if __name__ == "__main__":
    run_normal_form_analysis()
    run_extensive_form()
    run_agent_simulation()
    run_analysis_plots()        # <-- add this line