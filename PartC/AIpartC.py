import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from envs.simple_dungeonworld_env import DungeonMazeEnv, Directions

# =============================
# Q1: Supervised Learning Task
#   Multi-class classification on 80x80 RGB dungeon images
# =============================
# Load the dataset (assumes NPZ with keys 'X', 'Y')
data = np.load('dungeon_images_colour80.npz', allow_pickle=True)
X_img = data['X']  # shape: (N, 80, 80, 3)
y_img = data['Y']  # shape: (N,)

# Preprocess: convert to grayscale, downsample to 10x10, flatten
def preprocess_images(X):
    # rgb to grayscale
    gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])
    # downsample by block averaging
    small = gray.reshape(gray.shape[0], 80//10, 10, 80//10, 10).mean(axis=(2,4))
    return small.reshape(gray.shape[0], -1)

X_flat = preprocess_images(X_img)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_img, test_size=0.2, random_state=42, stratify=y_img
)

# Logistic Regression baseline
lr = LogisticRegression(max_iter=2000, multi_class='multinomial')
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, preds_lr)
print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
print(classification_report(y_test, preds_lr))

# Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, preds_rf)
print(f"Random Forest Accuracy:       {acc_rf:.4f}")
print(classification_report(y_test, preds_rf))

# =============================
# Q2: Unsupervised Learning Task
#   Cluster entities by sensorstats_partC.csv
# =============================
# Load sensor stats
df = pd.read_csv('dungeon_sensorstats_partC.csv')
features = ['stench', 'strength', 'heat', 'sound']
X_unsup = df[features].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unsup)

# Determine optimal k via elbow and silhouette
inertias, sils = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))

# Plot elbow & silhouette
def plot_metrics():
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, 11), inertias, 'bx-')
    plt.xlabel('k'); plt.ylabel('Inertia'); plt.title('Elbow Method')
    plt.subplot(1, 2, 2)
    plt.plot(range(2, 11), sils, 'bx-')
    plt.xlabel('k'); plt.ylabel('Silhouette Score'); plt.title('Silhouette Analysis')
    plt.show()

plot_metrics()
# Choose k_star from silhouette peak
k_star = int(range(2, 11)[np.argmax(sils)])
print(f"Chosen k (silhouette): {k_star}")

# Final clustering
km_final = KMeans(n_clusters=k_star, random_state=42).fit(X_scaled)
gmm = GaussianMixture(n_components=k_star, random_state=42).fit(X_scaled)
labels_km = km_final.predict(X_scaled)
labels_gmm = gmm.predict(X_scaled)

# 2D visualization via first two features (e.g., stench vs strength)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_km, cmap='tab10', alpha=0.6)
plt.title(f'KMeans (k={k_star})')
plt.xlabel(features[0]); plt.ylabel(features[1])

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_gmm, cmap='tab10', alpha=0.6)
plt.title(f'GMM (k={k_star})')
plt.xlabel(features[0]); plt.ylabel(features[1])
plt.show()

# =============================
# Q3: Reinforcement Learning Task
#   Q-learning on DungeonMazeEnv
# =============================
# Build state mapping
def build_state_mapping(env):
    idx_to_state, state_to_idx = [], {}
    size = env.grid_size
    for x in range(1, size-1):
        for y in range(1, size-1):
            for d in range(len(Directions)):
                idx = len(idx_to_state)
                idx_to_state.append((x, y, d))
                state_to_idx[((x, y), d)] = idx
    return idx_to_state, state_to_idx

# Q-learning implementation
def q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=1.0, eps_decay=0.9995):
    idx_to_state, st2i = build_state_mapping(env)
    nS, nA = len(idx_to_state), env.action_space.n
    Q = np.zeros((nS, nA))
    success_rates = []
    for ep in range(1, num_episodes+1):
        obs, _ = env.reset()
        s = st2i[(tuple(obs['robot_position']), obs['robot_direction'])]
        done = False
        total_r = 0
        while not done:
            if np.random.rand() < epsilon:
                a = np.random.choice(nA)
            else:
                a = np.argmax(Q[s])
            next_obs, r, term, trunc, _ = env.step(a)
            total_r += r
            ns = st2i[(tuple(next_obs['robot_position']), next_obs['robot_direction'])]
            Q[s, a] += alpha * (r + gamma * np.max(Q[ns]) - Q[s, a])
            s = ns
            done = term or trunc
        # track success
        success = 1 if r > 0 else 0
        success_rates.append(success)
        epsilon *= eps_decay
        if ep % 1000 == 0:
            rate = np.mean(success_rates[-1000:])
            print(f"Episode {ep}: Success rate (last 1000) = {rate:.3f}")
    # derive policy
    policy = np.argmax(Q, axis=1)
    return Q, policy, success_rates

# Run Q-learning on 8x8 grid
env_rl = DungeonMazeEnv(grid_size=8)
Q, policy_opt, succ = q_learning(env_rl)
print(f"Overall success rate: {np.mean(succ):.3f}")
