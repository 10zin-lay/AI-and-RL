import numpy as np
from collections import defaultdict
import gymnasium as gym

# ------------- Helper: Rollout and Policies (Q1) -------------

def rollout(env, policy_fn, max_steps=100):
    traj = []
    state, _ = env.reset()
    for _ in range(max_steps):
        action = policy_fn(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        traj.append((state, action, reward, next_state))
        state = next_state
        if done:
            break
    return traj

def random_policy(env):
    def pi(s):
        return env.action_space.sample()
    return pi

def all_forward_policy(env):
    def pi(s):
        return 0
    return pi

def custom_policy(env):
    def pi(s):
        if np.random.rand() < 0.7:
            return 0
        else:
            return env.action_space.sample()
    return pi

# ------------- Q2: Model-Based (Policy & Value Iteration) -------------

def policy_evaluation(policy, P, nS, nA, gamma, theta=1e-4):
    V = np.zeros(nS)
    for i in range(1000):  # Added hard iteration cap
        delta = 0
        for s in range(nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_s, reward, done in P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_s])
            delta = max(delta, abs(V[s] - v))
            V[s] = v
        print(f"[Policy Eval] Iter {i}, delta = {delta:.6f}", flush=True)
        if delta < theta:
            break
    return V

def policy_improvement(V, P, nS, nA, gamma):
    policy = np.zeros((nS, nA))
    for s in range(nS):
        Qs = np.zeros(nA)
        for a in range(nA):
            for prob, ns, r, done in P[s][a]:
                Qs[a] += prob * (r + gamma * V[ns])
        best_a = np.argmax(Qs)
        policy[s, best_a] = 1.0
    return policy

def policy_iteration(env, gamma=1.0):
    P = env.P
    nS = env.observation_space.n
    nA = env.action_space.n
    policy = np.ones((nS, nA)) / nA
    print("Starting Policy Iteration...", flush=True)
    for i in range(100):  # Limit iterations
        V = policy_evaluation(policy, P, nS, nA, gamma)
        new_policy = policy_improvement(V, P, nS, nA, gamma)
        if np.allclose(new_policy, policy):
            print(f"Policy iteration converged at iteration {i}", flush=True)
            break
        print(f"Policy iteration update {i}", flush=True)
        policy = new_policy
    return policy, V

def value_iteration(env, gamma=1.0, theta=1e-4):
    P = env.P
    nS = env.observation_space.n
    nA = env.action_space.n
    V = np.zeros(nS)
    print("Starting Value Iteration...", flush=True)
    for i in range(1000):  # Limit iterations
        delta = 0
        for s in range(nS):
            Qs = [
                sum(prob * (r + gamma * V[ns]) for prob, ns, r, done in P[s][a])
                for a in range(nA)
            ]
            max_q = max(Qs)
            delta = max(delta, abs(V[s] - max_q))
            V[s] = max_q
        print(f"[Value Iter] Iter {i}, delta = {delta:.6f}", flush=True)
        if delta < theta:
            break
    policy = np.zeros((nS, nA))
    for s in range(nS):
        Qs = [
            sum(prob * (r + gamma * V[ns]) for prob, ns, r, done in P[s][a])
            for a in range(nA)
        ]
        policy[s, np.argmax(Qs)] = 1.0
    return policy, V

# ------------- Q3: Model-Free (MC & TD) -------------

def mc_prediction(env, policy_fn, num_episodes=1000, gamma=0.99):
    nS = env.observation_space.n
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = np.zeros(nS)
    for ep in range(num_episodes):
        traj = rollout(env, policy_fn, max_steps=100)
        G = 0
        visited = set()
        for state, action, reward, next_state in reversed(traj):
            s_idx = state
            G = gamma * G + reward
            if s_idx not in visited:
                returns_sum[s_idx] += G
                returns_count[s_idx] += 1
                V[s_idx] = returns_sum[s_idx] / returns_count[s_idx]
                visited.add(s_idx)
        if (ep + 1) % 200 == 0:
            print(f"[MC] Episode {ep+1}", flush=True)
    return V

def td_prediction(env, policy_fn, num_episodes=1000, gamma=0.99, alpha=0.1):
    nS = env.observation_space.n
    V = np.zeros(nS)
    for ep in range(num_episodes):
        state, _ = env.reset()
        for _ in range(100):
            a = policy_fn(state)
            next_s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            V[state] += alpha * (r + gamma * V[next_s] - V[state])
            state = next_s
            if done:
                break
        if (ep + 1) % 200 == 0:
            print(f"[TD] Episode {ep+1}", flush=True)
    return V


# ------------- Putting It All Together -------------

if __name__ == "__main__":
    try:
        env = gym.make("FrozenLake-v1", is_slippery=False)
        env = env.unwrapped

        print("Has P:", hasattr(env, "P"), flush=True)

        print("Running simple policies...\n", flush=True)
        random_pi = random_policy(env)
        fw_pi     = all_forward_policy(env)
        cust_pi   = custom_policy(env)

        for name, pi in [("Random", random_pi), ("Forward", fw_pi), ("Custom", cust_pi)]:
            traj = rollout(env, pi)
            print(f"{name} policy trajectory length: {len(traj)}", flush=True)

        print("\nRunning Policy Iteration...\n", flush=True)
        pi_pi, v_pi = policy_iteration(env, gamma=1.0)
        print("Policy Iteration argmax policy:\n", np.argmax(pi_pi, axis=1).reshape(4, 4), flush=True)

        print("\nRunning Value Iteration...\n", flush=True)
        pi_vi, v_vi = value_iteration(env, gamma=1.0)
        print("Value Iteration argmax policy:\n", np.argmax(pi_vi, axis=1).reshape(4, 4), flush=True)

        print("\nRunning Monte Carlo Prediction...\n", flush=True)
        mc_V = mc_prediction(env, random_pi, num_episodes=2000)
        print("MC Value estimate:\n", mc_V.reshape((4, 4)), flush=True)

        print("\nRunning Temporal Difference Prediction...\n", flush=True)
        td_V = td_prediction(env, random_pi, num_episodes=2000, alpha=0.05)
        print("TD Value estimate:\n", td_V.reshape((4, 4)), flush=True)

    except Exception as e:
        print("Error occurred:", e, flush=True)
