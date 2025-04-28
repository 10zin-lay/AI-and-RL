import numpy as np
from collections import defaultdict
from envs.simple_dungeonworld_env import DungeonMazeEnv, Directions
import logging
import matplotlib.pyplot as plt  # for plotting

# ——— Configuration —————————————————————————————————————————————————
logging.basicConfig(level=logging.INFO, format="%(message)s")
GRID_SIZES = {"Q1": 6, "Q2": 8, "Q3": 10}
NUM_EPISODES = {"MC": 1000, "TD": 5000}
GAMMA = 0.99
ALPHA = 0.1
THETA = 1e-3
MAX_ITER = 20
MAX_STEPS = 100

# ——— State Mapping —————————————————————————————————————————————————
def build_state_mapping(env):
    """Compute once: bidirectional maps between idx and (x,y,dir)."""
    size = env.grid_size
    idx_to_state = []
    state_to_idx = {}
    for x in range(1, size - 1):
        for y in range(1, size - 1):
            for d in range(len(Directions)):
                idx = len(idx_to_state)
                idx_to_state.append((x, y, d))
                state_to_idx[((x, y), d)] = idx
    return idx_to_state, state_to_idx

# ——— Rollout & Policy Utilities ——————————————————————————————————————
def rollout(env, policy_fn, state_to_idx, max_steps=MAX_STEPS):
    """
    Follow policy_fn for up to max_steps.
    Returns trajectory list of (s,a,r) and success flag.
    """
    obs, _ = env.reset()
    s = state_to_idx[(tuple(obs["robot_position"]), obs["robot_direction"])]
    traj, total_reward = [], 0

    for _ in range(max_steps):
        a = policy_fn(s)
        next_obs, r, term, trunc, _ = env.step(a)
        total_reward += r
        traj.append((s, a, r))
        s = state_to_idx[(tuple(next_obs["robot_position"]), next_obs["robot_direction"])]
        if term or trunc:
            return traj, (r > 0)
    return traj, False

def make_random_policy(nA, p=[0.2,0.2,0.6]):
    return lambda s: np.random.choice(nA, p=p)

def make_all_forward_policy(forward_action=2):
    return lambda s: forward_action

def make_custom_policy(idx_to_state, grid_size):
    """
    Try forward; if blocked, choose left or right randomly.
    """
    def policy(s):
        x, y, d = idx_to_state[s]
        dx, dy = [(0,-1),(-1,0),(0,1),(1,0)][d]
        nx, ny = x+dx, y+dy
        if 1 <= nx < grid_size-1 and 1 <= ny < grid_size-1:
            return 2  # forward
        return np.random.choice([0,1])  # turn left/right
    return policy

# ——— Model‐Based Methods —————————————————————————————————————————
def build_transition_model(env, idx_to_state, state_to_idx):
    nA = env.action_space.n
    P = {s: {a: [] for a in range(nA)} for s in range(len(idx_to_state))}
    test_env = DungeonMazeEnv(grid_size=env.grid_size)

    for s, (x, y, d) in enumerate(idx_to_state):
        test_env.reset()
        test_env.robot_position = np.array([x, y])
        test_env.robot_direction = d
        for a in range(nA):
            obs, r, term, trunc, _ = test_env.step(a)
            done = term or trunc
            ns = state_to_idx[(tuple(obs["robot_position"]), obs["robot_direction"])]
            P[s][a] = [(1.0, ns, r, done)]
    return P

def policy_iteration(env, gamma=1.0, theta=THETA, max_iter=MAX_ITER):
    logging.info("=== Policy Iteration ===")
    idx_to_state, st2i = build_state_mapping(env)
    P = build_transition_model(env, idx_to_state, st2i)
    nS, nA = len(idx_to_state), env.action_space.n
    policy = np.ones((nS, nA)) / nA
    deltas = []

    for i in range(max_iter):
        # evaluation
        V = np.zeros(nS)
        for _ in range(1000):
            delta = 0
            for s in range(nS):
                v = sum(policy[s,a] * sum(p*(r + gamma*V[ns]) for p,ns,r,_ in P[s][a])
                        for a in range(nA))
                delta = max(delta, abs(V[s] - v))
                V[s] = v
            if delta < theta:
                break
        deltas.append(delta)
        logging.info(f"PI Iter {i+1}, delta={delta:.5f}")

        # improvement
        new_policy = np.zeros((nS, nA))
        for s in range(nS):
            qs = [sum(p*(r + gamma*V[ns]) for p,ns,r,_ in P[s][a]) for a in range(nA)]
            new_policy[s, np.argmax(qs)] = 1.0

        if np.allclose(policy, new_policy):
            logging.info(f"Policy converged at iteration {i+1}")
            break
        policy = new_policy

    return policy, V, deltas

def value_iteration(env, gamma=1.0, theta=THETA, max_iter=MAX_ITER):
    logging.info("=== Value Iteration ===")
    idx_to_state, st2i = build_state_mapping(env)
    P = build_transition_model(env, idx_to_state, st2i)
    nS, nA = len(idx_to_state), env.action_space.n
    V = np.zeros(nS)
    deltas = []

    for i in range(max_iter):
        delta = 0
        for s in range(nS):
            qs = [sum(p*(r + gamma*V[ns]) for p,ns,r,_ in P[s][a]) for a in range(nA)]
            max_v = max(qs)
            delta = max(delta, abs(V[s] - max_v))
            V[s] = max_v
        deltas.append(delta)
        logging.info(f"VI Iter {i+1}, delta={delta:.5f}")
        if delta < theta:
            break

    policy = np.zeros((nS, nA))
    for s in range(nS):
        qs = [sum(p*(r + gamma*V[ns]) for p,ns,r,_ in P[s][a]) for a in range(nA)]
        policy[s, np.argmax(qs)] = 1.0

    return policy, V, deltas

# ——— Model‐Free Predictions —————————————————————————————————————————
def mc_prediction(env, policy_fn, num_episodes=NUM_EPISODES['MC'], gamma=GAMMA):
    logging.info("=== Monte Carlo Prediction ===")
    idx_to_state, st2i = build_state_mapping(env)
    returns, counts = defaultdict(float), defaultdict(int)
    V = np.zeros(len(idx_to_state))
    coverage = []

    for ep in range(1, num_episodes+1):
        traj, _ = rollout(env, policy_fn, st2i)
        G = 0; visited = set()
        for s,a,r in reversed(traj):
            G = gamma * G + r
            if s not in visited:
                returns[s] += G
                counts[s] += 1
                V[s] = returns[s] / counts[s]
                visited.add(s)
        coverage.append(len(visited)/len(idx_to_state))
        if ep % 100 == 0:
            logging.info(f"Episode {ep}: visited {coverage[-1]*100:.1f}% states")

    return V, coverage

def td_prediction(env, policy_fn, num_episodes=NUM_EPISODES['TD'],
                  gamma=GAMMA, alpha=ALPHA):
    logging.info("=== TD(0) Prediction ===")
    idx_to_state, st2i = build_state_mapping(env)
    V = np.zeros(len(idx_to_state))
    coverage = []

    for ep in range(1, num_episodes+1):
        obs,_ = env.reset()
        s = st2i[(tuple(obs["robot_position"]), obs["robot_direction"])]
        visited = {s}
        done = False
        while not done:
            a = policy_fn(s)
            next_obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            s_next = st2i[(tuple(next_obs["robot_position"]), next_obs["robot_direction"])]
            V[s] += alpha*(r + gamma*V[s_next] - V[s])
            s = s_next
            visited.add(s)
        coverage.append(len(visited)/len(idx_to_state))
        if ep % 500 == 0:
            logging.info(f"Episode {ep}: visited {coverage[-1]*100:.1f}% states")

    return V, coverage

# ——— Main Execution: Q1/Q2/Q3 ————————————————————————————————————————
if __name__ == "__main__":
    # Q1: Policy evaluation statistics
    env1 = DungeonMazeEnv(grid_size=GRID_SIZES['Q1'])
    idx_to_state1, st2i1 = build_state_mapping(env1)
    policies = {
        "Random": make_random_policy(env1.action_space.n),
        "Forward": make_all_forward_policy(),
        "Custom": make_custom_policy(idx_to_state1, env1.grid_size),
    }
    for name, fn in policies.items():
        traj, success = rollout(env1, fn, st2i1)
        logging.info(f"{name} → len={len(traj)}, total_r={sum(r for _,_,r in traj)}, success={success}")

    # Q2: Model-based
    env2 = DungeonMazeEnv(grid_size=GRID_SIZES['Q2'])
    pi_policy, pi_V, pi_d = policy_iteration(env2)
    vi_policy, vi_V, vi_d = value_iteration(env2)

    # Q3: Model-free
    env3 = DungeonMazeEnv(grid_size=GRID_SIZES['Q3'])
    mc_V, mc_cov = mc_prediction(env3, make_random_policy(env3.action_space.n))
    td_V, td_cov = td_prediction(env3, make_random_policy(env3.action_space.n))

    # Expose convergence data for plotting
    # Plot PI vs VI Delta convergence
    plt.figure()
    plt.plot(pi_d, label='Policy Iteration Δ')
    plt.plot(vi_d, label='Value Iteration Δ')
    plt.xlabel('Iteration')
    plt.ylabel('Max Δ')
    plt.title('Convergence of PI and VI')
    plt.legend()
    plt.savefig('pi_vi_convergence.png', dpi=300)
    plt.close()

    # Plot MC vs TD coverage over episodes
    plt.figure()
    plt.plot(mc_cov, label='MC Coverage')
    plt.plot(td_cov, label='TD Coverage')
    plt.xlabel('Episode')
    plt.ylabel('Fraction of States Visited')
    plt.title('State Coverage over Episodes')
    plt.legend()
    plt.savefig('mc_td_coverage.png', dpi=300)
    plt.close()
