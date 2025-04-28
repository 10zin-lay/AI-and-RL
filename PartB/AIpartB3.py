import numpy as np
from collections import defaultdict

from envs.simple_dungeonworld_env import DungeonMazeEnv, Actions, Directions

# ─── Q1 Helpers ────────────────────────────────────────────────────────────────
def build_state_mapping(env):
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
    # Weighted random policy: 20% up, 20% left, 60% down (as per report)
    return lambda s: np.random.choice([0, 1, 2], p=[0.2, 0.2, 0.6])

def all_forward_policy(env):
    # Always move down (action 2)
    return lambda s: 2

def custom_policy(env):
    # Wall avoidance with forward bias (as described in report)
    def policy(s):
        # Try simulating a forward move
        temp_env = DungeonMazeEnv(render_mode=None, grid_size=env.grid_size)
        temp_env.reset()
        # Set to same state as original environment
        idx_to_state, _ = build_state_mapping(env)
        temp_env.robot_position = np.array(idx_to_state[s][:2])
        temp_env.robot_direction = idx_to_state[s][2]
        
        # Try forward move
        _, r, _, _, _ = temp_env.step(2)  # Down/forward action
        
        if r <= -10:  # Wall detected (adjust threshold as needed)
            # Turn left or right randomly
            return np.random.choice([0, 1])  # Up or Left
        else:
            # Move forward
            return 2  # Down
    
    return policy


# ─── Transition Model for Q2 ──────────────────────────────────────────────────
def build_transition_model(env, idx_to_state, state_to_idx):
    nA = env.action_space.n
    P = {s: {a: [] for a in range(nA)} for s in range(len(idx_to_state))}
    
    # Create a copy of the environment for transition modeling
    model_env = DungeonMazeEnv(render_mode=None, grid_size=env.grid_size)
    
    for s_idx, (x, y, d) in enumerate(idx_to_state):
        for a in range(nA):
            # Reset environment for each transition
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
def policy_evaluation(policy, P, nS, nA, gamma, theta=1e-4, max_iter=1000):
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
            break
    return V

def policy_improvement(V, P, nS, nA, gamma):
    policy = np.zeros((nS, nA))
    for s in range(nS):
        Qs = np.array([
            sum(prob * (r + gamma * V[ns]) for prob, ns, r, done in P[s][a])
            for a in range(nA)
        ])
        policy[s, np.argmax(Qs)] = 1.0
    return policy

def policy_iteration(env, gamma=1.0, theta=1e-4):
    idx_to_state, state_to_idx = build_state_mapping(env)
    P = build_transition_model(env, idx_to_state, state_to_idx)
    nS, nA = len(idx_to_state), env.action_space.n
    policy = np.ones((nS, nA)) / nA
    
    for i in range(100):  # Maximum iterations
        V = policy_evaluation(policy, P, nS, nA, gamma, theta)
        new_policy = policy_improvement(V, P, nS, nA, gamma)
        if np.allclose(new_policy, policy):
            print(f"Policy iteration converged after {i+1} iterations")
            break
        policy = new_policy
    
    return policy, V

def value_iteration(env, gamma=1.0, theta=1e-4, max_iter=1000):
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
    G = 0
    returns = []
    for _, _, r in reversed(traj):
        G = gamma * G + r
        returns.insert(0, G)
    return returns

def mc_prediction(env, policy_fn, num_episodes=1000, gamma=0.99):
    idx_to_state, state_to_idx = build_state_mapping(env)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = np.zeros(len(idx_to_state))
    
    for ep in range(num_episodes):
        traj = rollout(env, policy_fn, state_to_idx, max_steps=100)
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

def td_prediction(env, policy_fn, gamma=0.99, alpha=0.1, num_episodes=5000):
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


# ─── Main: run Q1–Q3 and aggregate over directions ─────────────────────────────
if __name__ == "__main__":
    # Q1: 6×6
    env1 = DungeonMazeEnv(render_mode=None, grid_size=6)
    _, st1 = build_state_mapping(env1)
    for name, pi in [("Random", random_policy(env1)),
                     ("Forward", all_forward_policy(env1)),
                     ("Custom", custom_policy(env1))]:
        traj = rollout(env1, pi, st1)
        print(f"{name} policy trajectory length (6×6): {len(traj)}")

    # Q2: model-based on 8×8
    env2 = DungeonMazeEnv(render_mode=None, grid_size=8)
    pi_pi, v_pi = policy_iteration(env2, gamma=1.0)
    best_pi = np.argmax(pi_pi, axis=1)
    # aggregate across the 4 directions for each cell
    n_dirs = len(Directions)
    n_cells = best_pi.size // n_dirs
    side = env2.grid_size - 2
    cell_actions = best_pi.reshape(n_cells, n_dirs).mean(axis=1).round().astype(int)
    print("Policy Iteration (cell grid):")
    print(cell_actions.reshape(side, side))

    pi_vi, v_vi = value_iteration(env2, gamma=1.0)
    best_vi = np.argmax(pi_vi, axis=1)
    cell_actions_vi = best_vi.reshape(n_cells, n_dirs).mean(axis=1).round().astype(int)
    print("Value Iteration (cell grid):")
    print(cell_actions_vi.reshape(side, side))

    # Q3: model-free on 10×10
    env3 = DungeonMazeEnv(render_mode=None, grid_size=10)
    print("Starting Monte Carlo prediction (this may take a while)...")
    mc_V = mc_prediction(env3, random_policy(env3), num_episodes=1000, gamma=0.99)
    print("Starting TD prediction (this may take a while)...")
    td_V = td_prediction(env3, random_policy(env3), gamma=0.99, alpha=0.1, num_episodes=5000)

    # aggregate values across directions
    n_dirs3 = len(Directions)
    n_cells3 = mc_V.size // n_dirs3
    side3 = env3.grid_size - 2

    cell_V_mc = mc_V.reshape(n_cells3, n_dirs3).mean(axis=1)
    print("MC Value (cell grid):")
    print(cell_V_mc.reshape(side3, side3))

    cell_V_td = td_V.reshape(n_cells3, n_dirs3).mean(axis=1)
    print("TD Value (cell grid):")
    print(cell_V_td.reshape(side3, side3))