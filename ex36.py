import numpy as np
from typing import Tuple
import ex34
import ex35
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List


def from2to1(i1: int, i2: int) -> int:
    return i1 * 21 + i2


def from1to2(s: int) -> Tuple[int, int]:
    return s // 21, s % 21


def makeTransProb_and_Rewards(
    shop1_trans: np.ndarray,  # 2D Array of floats (transition probabilities)
    shop2_trans: np.ndarray,  # 2D Array of floats (transition probabilities)
    shop1_rew: np.ndarray,   # 1D Array of floats (expected rewards)
    shop2_rew: np.ndarray    # 1D Array of floats (expected rewards)
    # Returns (3D Array of floats, 2D Array of floats, 2D Array of bools)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nStates: int = 21 * 21
    nActions: int = 11

    probStates: np.ndarray = np.zeros(
        (nStates, nActions, nStates))  # 3D Array of floats
    reward: np.ndarray = np.zeros(
        (nStates, nActions))              # 2D Array of floats

    allowedActions: np.ndarray = np.zeros((nStates, nActions), dtype=bool)

    for s in range(nStates):
        i1, i2 = from1to2(s)
        for a_idx, a in enumerate(range(-5, 6)):
            if 0 <= i1 + a <= 20 and 0 <= i2 - a <= 20:
                allowedActions[s, a_idx] = True
                reward[s, a_idx] = ex35.get_total_expected_reward(
                    shop1_rew, shop2_rew, i1, i2, a)

                for s_next in range(nStates):
                    j1, j2 = from1to2(s_next)
                    prob = ex35.get_joint_trans_prob(
                        shop1_trans, shop2_trans, i1, i2, j1, j2, a)
                    probStates[s, a_idx, s_next] = prob

    return probStates, reward, allowedActions


if __name__ == '__main__':
    # Setup basic parameters
    gamma: float = 0.9
    theta: float = 1e-4  # Convergence threshold for policy evaluation
    nStates: int = 441
    nActions: int = 11

    # Generate the base arrays from ex35
    shop1_trans: np.ndarray = ex35.oneTransProb(3.0, 3.0)  # 2D Array of floats
    shop2_trans: np.ndarray = ex35.oneTransProb(4.0, 2.0)  # 2D Array of floats
    shop1_rew: np.ndarray = ex35.oneReward(3.0)           # 1D Array of floats
    shop2_rew: np.ndarray = ex35.oneReward(4.0)           # 1D Array of floats

    probStates, reward, allowedActions = makeTransProb_and_Rewards(
        shop1_trans, shop2_trans, shop1_rew, shop2_rew
    )

    # Initialize deterministic policy
    probActions: np.ndarray = np.zeros(
        (nStates, nActions))  # 2D Array of floats
    probActions[:, 5] = 1.0  # Index 5 is action 0 (no cars moved)

    # Initialize value function
    v: np.ndarray = np.zeros(nStates)  # 1D Array of floats

    policies: List[np.ndarray] = []

    # Save the initial policy (Iteration 0)
    initial_policy: np.ndarray = np.argmax(probActions, axis=1) - 5
    policies.append(initial_policy.reshape((21, 21)))

    policy_stable: bool = False
    iteration: int = 0

    while not policy_stable:
        print(f"Iteration {iteration}")

        # 1. Policy Evaluation
        while True:
            v_new: np.ndarray = ex34.policy_evaluation_step(
                v, probActions, probStates, reward, gamma)
            if float(np.max(np.abs(v - v_new))) < theta:
                v = np.copy(v_new)
                break
            v = np.copy(v_new)

        # 2. Policy Improvement
        new_probActions: np.ndarray = ex34.derive_greedy_policy(
            v, probStates, reward, gamma, allowedActions)

        # Save the updated policy
        current_policy: np.ndarray = np.argmax(new_probActions, axis=1) - 5
        policies.append(current_policy.reshape((21, 21)))

        # 3. Check for convergence
        if np.array_equal(probActions, new_probActions):
            policy_stable = True
        else:
            probActions = np.copy(new_probActions)
            iteration += 1

    # Create a figure with a 2x3 grid to match the textbook's layout
    fig = plt.figure(figsize=(15, 10))

    # 1. Plot the first 5 policies (pi_0 through pi_4) with text numbers
    for i, pol in enumerate(policies):
        if i > 4:
            break  # Figure 4.2 only shows the first 5 iterations

        ax = fig.add_subplot(2, 3, i + 1)

        # Set axis limits to neatly frame the 21x21 grid
        ax.set_xlim(-0.5, 20.5)
        ax.set_ylim(-0.5, 20.5)

        ax.set_title(f'$\\pi_{i}$')
        ax.set_xlabel('#Cars at second location')
        ax.set_ylabel('#Cars at first location')
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_yticks([0, 5, 10, 15, 20])

        # Optional: add a light grid to help guide the eye across the numbers
        ax.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)

        # Iterate through the grid and write the action value at each coordinate
        for y in range(21):     # y corresponds to i1 (first location)
            for x in range(21):  # x corresponds to i2 (second location)
                action_value = -int(pol[y, x])

                # To make the plot less cluttered, you can skip rendering the zeros
                text_str = str(action_value) if action_value != 0 else "."

                ax.text(x, y, text_str, ha='center', va='center', fontsize=6)

    # 2. Plot the final state-value function (v_pi_4) as a 3D surface
    ax_v = fig.add_subplot(2, 3, 6, projection='3d')

    # Create a meshgrid for the 21x21 state space
    # X represents the second location, Y represents the first location
    X, Y = np.meshgrid(range(21), range(21))
    Z: np.ndarray = v.reshape((21, 21))

    ax_v.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax_v.set_title('$v_{\\pi_4}$')
    ax_v.set_xlabel('#Cars at second location')
    ax_v.set_ylabel('#Cars at first location')

    plt.tight_layout()
    plt.show()
