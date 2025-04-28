import numpy as np
from collections import defaultdict
from envs.simple_dungeonworld_env import DungeonMazeEnv, Actions, Directions

# ─── Configuration Parameters ─────────────────────────────────────────────────
GRID_SIZES = {'Q1': 6, 'Q2': 8, 'Q3': 10}
NUM_EPISODES = {'MC': 1000, 'TD': 5000}
GAMMA = 0.99
ALPHA = 0.1
THETA = 1e-4
MAX_ITER = 1000

# ─── Q1 Helpers ────────────────────────────────────────────────────────────────
def build_state_mapping(env):
    """Build bidirectional mapping between state indices and (position, direction) tuples."""
    obs, _ = env.reset()
    size = env.grid_size
    idx_to_state = []
    state_to_idx = {}
    for x in range(1, size-1):
        for y in range(1, size-1):
            for d in range(len(Directions)):
                idx = len(idx_to_state)
                idx_to_state.append((x, y, d))
                state_to_idx[((x, y), d)] = idx
    return idx_to_state, state_to_idx

def rollout(env, policy_fn, state_to_idx, max_steps=100):
    """Generate trajectory by following given policy in the environment."""
    traj = []
    obs, _ = env.reset()
    s = state_to_idx[(tuple(obs["robot_position"]), obs["robot_direction"])]
    
    for _ in range(max_steps):
        a = policy_fn(s)
        next_obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        s_next = state_to_idx[(tuple(next_obs["robot_position"]), next_obs["robot_direction"])]
        traj.append((s, a, r))
        s = s_next
        if done:
            break
    return traj

def random_policy(env):
    """Random policy with weighted action probabilities."""
    return lambda s: np.random.choice([0, 1, 2], p=[0.2, 0.2, 0.6])

def all_forward_policy(env):
    """Deterministic policy that always moves forward."""
    return lambda s: 2

def custom_policy(env):
    """Custom policy that avoids walls with forward bias."""
    def policy(s):
        idx_to_state, _ = build_state_mapping(env)
        x, y, d = idx_to_state[s]
        
        # Simulate forward move
        new_x, new_y = x, y
        if d == 0:    # Up
            new_y -= 1
        elif d == 1:  # Left
            new_x -= 1
        elif d == 2:  # Down
            new_y += 1
        elif d == 3:  # Right
            new_x += 1
            
        # Check if new position is valid (not a wall)
        if (1 <= new_x < env.grid_size-1) and (1 <= new_y < env.grid_size-1):
            return 2  # Forward
        else:
            return np.random.choice([0, 1])  # Turn if wall ahead
    return policy

# ─── Transition Model for Q2 ──────────────────────────────────────────────────
def build_transition_model(env, idx_to_state, state_to_idx):
    """Build transition model P(s'|s,a) for the environment."""
    nA = env.action_space.n
    P = {s: {a: [] for a in range(nA)} for s in range(len(idx_to_state))}
    
    model_env = DungeonMazeEnv(render_mode=None, grid_size=env.grid_size)
    
    for s_idx, (x, y, d) in enumerate(idx_to_state):
        for a in range(nA):
            model_env.reset()
            model_env.robot_position = np.array([x, y])
            model_env.robot_direction = d
            
            next_obs, r, term, trunc, _ = model_env.step(a)
            done = term or trunc
            nx, ny = next_obs["robot_position"]
            nd = next_obs["robot_direction"]
            ns_idx = state_to_idx[((nx, ny), nd)]
            P[s_idx][a] = [(1.0, ns_idx, r, done)]
    
    return P

# ─── Q2 Model-Based Methods ───────────────────────────────────────────────────
def policy_evaluation(policy, P, nS, nA, gamma, theta=THETA, max_iter=MAX_ITER):
    """Evaluate policy using iterative policy evaluation."""
    V = np.zeros(nS)
    for i in range(max_iter):
        delta = 0
        for s in range(nS):
            v = sum(policy[s, a] * prob * (r + gamma * V[ns])
                    for a in range(nA)
                    for prob, ns, r, done in P[s][a])
            delta = max(delta, abs(V[s] - v))
            V[s] = v
        if delta < theta:
            print(f"Policy evaluation converged after {i+1} iterations")
            break
    return V

def policy_improvement(V, P, nS, nA, gamma):
    """Improve policy by being greedy wrt current value function."""
    policy = np.zeros((nS, nA))
    for s in range(nS):
        Qs = np.array([
            sum(prob * (r + gamma * V[ns]) for prob, ns, r, done in P[s][a])
            for a in range(nA)
        ])
        policy[s, np.argmax(Qs)] = 1.0
    return policy

def policy_iteration(env, gamma=1.0, theta=THETA):
    """Perform policy iteration to find optimal policy."""
    idx_to_state, state_to_idx = build_state_mapping(env)
    P = build_transition_model(env, idx_to_state, state_to_idx)
    nS, nA = len(idx_to_state), env.action_space.n
    policy = np.ones((nS, nA)) / nA
    
    for i in range(MAX_ITER):
        V = policy_evaluation(policy, P, nS, nA, gamma, theta)
        new_policy = policy_improvement(V, P, nS, nA, gamma)
        
        if np.allclose(new_policy, policy):
            print(f"Policy iteration converged after {i+1} iterations")
            break
        policy = new_policy
    
    return policy, V

def value_iteration(env, gamma=1.0, theta=THETA, max_iter=MAX_ITER):
    """Perform value iteration to find optimal policy."""
    idx_to_state, state_to_idx = build_state_mapping(env)
    P = build_transition_model(env, idx_to_state, state_to_idx)
    nS, nA = len(idx_to_state), env.action_space.n
    V = np.zeros(nS)
    
    for i in range(max_iter):
        delta = 0
        for s in range(nS):
            Qs = [sum(prob * (r + gamma * V[ns]) for prob, ns, r, done in P[s][a])
                  for a in range(nA)]
            m = max(Qs)
            delta = max(delta, abs(V[s] - m))
            V[s] = m
        if delta < theta:
            print(f"Value iteration converged after {i+1} iterations")
            break
    
    policy = np.zeros((nS, nA))
    for s in range(nS):
        Qs = [sum(prob * (r + gamma * V[ns]) for prob, ns, r, done in P[s][a])
              for a in range(nA)]
        policy[s, np.argmax(Qs)] = 1.0
    
    return policy, V

# ─── Q3 Model-Free Methods ─────────────────────────────────────────────────────
def compute_returns(traj, gamma):
    """Calculate discounted returns for a trajectory."""
    G = 0
    returns = []
    for _, _, r in reversed(traj):
        G = gamma * G + r
        returns.insert(0, G)
    return returns

def mc_prediction(env, policy_fn, num_episodes=NUM_EPISODES['MC'], gamma=GAMMA):
    """Estimate value function using Monte Carlo prediction."""
    idx_to_state, state_to_idx = build_state_mapping(env)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = np.zeros(len(idx_to_state))
    
    for ep in range(num_episodes):
        traj = rollout(env, policy_fn, state_to_idx)
        R = compute_returns(traj, gamma)
        visited = set()
        
        for (s, _, _), G in zip(traj, R):
            if s not in visited:  # First-visit MC
                returns_sum[s] += G
                returns_count[s] += 1
                V[s] = returns_sum[s] / returns_count[s]
                visited.add(s)
        
        if (ep + 1) % 100 == 0:
            print(f"MC: Completed {ep + 1}/{num_episodes} episodes")
    
    return V

def td_prediction(env, policy_fn, gamma=GAMMA, alpha=ALPHA, num_episodes=NUM_EPISODES['TD']):
    """Estimate value function using Temporal Difference learning."""
    idx_to_state, state_to_idx = build_state_mapping(env)
    V = np.zeros(len(idx_to_state))
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        s = state_to_idx[(tuple(obs["robot_position"]), obs["robot_direction"])]
        done = False
        
        while not done:
            a = policy_fn(s)
            next_obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            s_next = state_to_idx[(tuple(next_obs["robot_position"]), next_obs["robot_direction"])]
            
            # TD update
            V[s] += alpha * (r + gamma * V[s_next] - V[s])
            s = s_next
        
        if (ep + 1) % 500 == 0:
            print(f"TD: Completed {ep + 1}/{num_episodes} episodes")
    
    return V

# ─── Analysis Utilities ───────────────────────────────────────────────────────
def analyze_policy(policy, idx_to_state):
    """Analyze and visualize the learned policy."""
    n_dirs = len(Directions)
    n_cells = policy.shape[0] // n_dirs
    side = int(np.sqrt(n_cells))
    
    # Aggregate actions across directions
    best_actions = np.argmax(policy, axis=1)
    cell_actions = best_actions.reshape(n_cells, n_dirs)
    
    # Find most common action for each cell
    from scipy.stats import mode
    cell_mode_actions = mode(cell_actions, axis=1)[0].flatten()
    
    print("Policy (most common action per cell):")
    print(cell_mode_actions.reshape(side, side))

def compare_value_functions(V1, V2, idx_to_state):
    """Compare two value functions."""
    n_dirs = len(Directions)
    n_cells = len(V1) // n_dirs
    
    # Aggregate values across directions
    V1_cells = V1.reshape(n_cells, n_dirs).mean(axis=1)
    V2_cells = V2.reshape(n_cells, n_dirs).mean(axis=1)
    
    # Calculate correlation
    corr = np.corrcoef(V1_cells, V2_cells)[0,1]
    print(f"Value function correlation: {corr:.3f}")
    
    # Calculate mean absolute difference
    mad = np.mean(np.abs(V1_cells - V2_cells))
    print(f"Mean absolute difference: {mad:.3f}")

# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Question 1: Policy Evaluation ===")
    env1 = DungeonMazeEnv(render_mode=None, grid_size=GRID_SIZES['Q1'])
    _, st1 = build_state_mapping(env1)
    
    policies = {
        "Random": random_policy(env1),
        "Forward": all_forward_policy(env1),
        "Custom": custom_policy(env1)
    }
    
    for name, policy in policies.items():
        traj = rollout(env1, policy, st1)
        rewards = [r for (_,_,r) in traj]
        print(f"{name} policy: Length={len(traj)}, Total Reward={sum(rewards)}")
        print(f"Reward sequence: {rewards}")

    print("\n=== Question 2: Model-Based Methods ===")
    env2 = DungeonMazeEnv(render_mode=None, grid_size=GRID_SIZES['Q2'])
    
    print("\nRunning Policy Iteration...")
    pi_pi, v_pi = policy_iteration(env2, gamma=1.0)
    analyze_policy(pi_pi, build_state_mapping(env2)[0])
    
    print("\nRunning Value Iteration...")
    pi_vi, v_vi = value_iteration(env2, gamma=1.0)
    analyze_policy(pi_vi, build_state_mapping(env2)[0])
    
    # Compare policies
    agreement = np.mean(np.argmax(pi_pi, axis=1) == np.argmax(pi_vi, axis=1))
    print(f"\nPolicy agreement between PI and VI: {agreement:.1%}")
    compare_value_functions(v_pi, v_vi, build_state_mapping(env2)[0])

    print("\n=== Question 3: Model-Free Methods ===")
    env3 = DungeonMazeEnv(render_mode=None, grid_size=GRID_SIZES['Q3'])
    
    print("\nRunning Monte Carlo Prediction...")
    mc_V = mc_prediction(env3, random_policy(env3))
    
    print("\nRunning TD Prediction...")
    td_V = td_prediction(env3, random_policy(env3))
    
    # Compare value functions
    compare_value_functions(mc_V, td_V, build_state_mapping(env3)[0])
    
    # Visualize value functions
    n_dirs = len(Directions)
    n_cells = len(mc_V) // n_dirs
    side = env3.grid_size - 2
    
    print("\nMC Value Estimates (averaged over directions):")
    print(mc_V.reshape(n_cells, n_dirs).mean(axis=1).reshape(side, side))
    
    print("\nTD Value Estimates (averaged over directions):")
    print(td_V.reshape(n_cells, n_dirs).mean(axis=1).reshape(side, side))