import numpy as np
from scipy.stats import poisson


def oneTransProb(lamRequest: float, lamReturn: float) -> np.ndarray:  # Returns 2D Array of floats
    n1States: int = 21
    # 2D Array of floats (transition probabilities)
    p: np.ndarray = np.zeros((n1States, n1States))

    for k0 in range(n1States):
        for k1 in range(n1States):
            x_min = max(0, k0 - k1)
            # 1D Array of integers (number of cars)
            x: np.ndarray = np.arange(x_min, k0 + 1)

            # 1D Array of floats (probabilities)
            p1: np.ndarray = poisson.pmf(x, lamRequest)
            if len(x) > 0:
                p1[-1] = 1.0 - poisson.cdf(x[-1] - 1, lamRequest)

            if k1 < (n1States - 1):
                # 1D Array of floats (probabilities)
                p2: np.ndarray = poisson.pmf(k1 - (k0 - x), lamReturn)
            else:
                p2: np.ndarray = 1.0 - \
                    poisson.cdf(k1 - (k0 - x) - 1, lamReturn)

            p[k0, k1] = np.sum(p1 * p2)

    return p


def oneReward(lamRequest: float) -> np.ndarray:  # Returns 1D Array of floats
    n1States: int = 21
    r: np.ndarray = np.zeros(n1States)  # 1D Array of floats (expected rewards)

    for k0 in range(n1States):
        # 1D Array of integers (number of cars)
        x: np.ndarray = np.arange(0, k0 + 1)
        # 1D Array of floats (probabilities)
        p1: np.ndarray = poisson.pmf(x, lamRequest)
        if len(x) > 0:
            p1[-1] = 1.0 - poisson.cdf(x[-1] - 1, lamRequest)

        r[k0] = np.sum(x * p1)

    return r


def get_joint_trans_prob(
    shop1_trans: np.ndarray,  # 2D Array of floats
    shop2_trans: np.ndarray,  # 2D Array of floats
    i1: int,
    i2: int,
    j1: int,
    j2: int,
    a: int
) -> float:
    if abs(a) > 5 or not (0 <= i1 + a <= 20 and 0 <= i2 - a <= 20):
        return 0.0

    prob_shop1: float = shop1_trans[i1 + a, j1]
    prob_shop2: float = shop2_trans[i2 - a, j2]

    return float(prob_shop1 * prob_shop2)


def get_total_expected_reward(
    shop1_reward: np.ndarray,  # 1D Array of floats
    shop2_reward: np.ndarray,  # 1D Array of floats
    i1: int,
    i2: int,
    a: int
) -> float:
    if abs(a) > 5 or not (0 <= i1 + a <= 20 and 0 <= i2 - a <= 20):
        return float('-inf')

    reward_shop1: float = 10.0 * shop1_reward[i1 + a]
    reward_shop2: float = 10.0 * shop2_reward[i2 - a]

    moving_cost: float = 2.0 * abs(a)

    return float(reward_shop1 + reward_shop2 - moving_cost)
