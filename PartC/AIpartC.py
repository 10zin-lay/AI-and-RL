### Part C: Code for Supervised, Unsupervised, and Reinforcement Learning Tasks

# 1. Supervised Learning: CNN for 5-class classification on 80x80 RGB dungeon images

import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Adjust to your data directory
DATA_DIR = "dungeon_images_colour80/"
BATCH_SIZE = 32
IMG_SIZE = (80, 80)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=123
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'val'),
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

# Normalize
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Model definition
num_classes = 5
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=IMG_SIZE + (3,)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax'),
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

# Save model
model.save('cnn_dungeon_classifier.h5')


# 2. Unsupervised Learning: Clustering on dungeon_sensorstats_partC.csv

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('dungeon_sensorstats_partC.csv')
X = df.drop(columns=['entity_id'])  # keep only numeric features

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Find optimal k via silhouette
sil_scores = []
K_RANGE = range(2, 11)
for k in K_RANGE:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Plot silhouette vs k
plt.figure(); plt.plot(list(K_RANGE), sil_scores, 'o-');
plt.xlabel('k'); plt.ylabel('Silhouette Score'); plt.title('GMM Silhouette'); plt.show()

# Fit final GMM
best_k = K_RANGE[sil_scores.index(max(sil_scores))]
gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42)
clusters = gmm.fit_predict(X_scaled)

# 2D scatter
plt.figure(figsize=(6,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='tab10', s=5)
plt.title(f'GMM clustering (k={best_k})'); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.show()


# 3. Reinforcement Learning: DQN for MazeDungeonWorld

import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Import your custom environment (ensure PYTHONPATH points to it)
from core.dungeonworld_grid import DungeonMazeEnv

# Hyperparameters
GRID_SIZE = 12
STATE_DIM = 4 * GRID_SIZE * GRID_SIZE
ACTION_DIM = 4  # e.g. [turn_left, turn_right, forward, no-op]
LR = 1e-3
GAMMA = 0.99
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 5000
BATCH_SIZE = 64
BUFFER_SIZE = 100000
TARGET_UPDATE = 1000
NUM_EPISODES = 2000
MAX_STEPS = 200

# Q-network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Replay Buffer
def make_epsilon(steps_done):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)

class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, *args): self.buffer.append(tuple(args))
    def sample(self, batch_size): return random.sample(self.buffer, batch_size)
    def __len__(self): return len(self.buffer)

# State encoder: one-hot to vector

def obs_to_state_vec(obs, grid_size):
    s = DungeonMazeEnv.obs_to_state(obs, grid_size)
    vec = np.zeros(grid_size*grid_size*4, dtype=np.float32)
    vec[s] = 1.0
    return vec

# Main training loop

env = DungeonMazeEnv(render_mode=None, grid_size=GRID_SIZE)
policy_net = DQN(STATE_DIM, ACTION_DIM)
target_net = DQN(STATE_DIM, ACTION_DIM)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer(BUFFER_SIZE)
steps_done = 0

for episode in range(1, NUM_EPISODES+1):
    obs, _ = env.reset()
    state = obs_to_state_vec(obs, GRID_SIZE)
    total_reward = 0
    for t in range(MAX_STEPS):
        eps = make_epsilon(steps_done)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = policy_net(torch.from_numpy(state).unsqueeze(0))
                action = q_vals.argmax().item()
        next_obs, reward, term, trunc, _ = env.step(action)
        next_state = obs_to_state_vec(next_obs, GRID_SIZE)
        done = term or trunc
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps_done += 1

        # Learn
        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = map(lambda x: torch.tensor(x), zip(*batch))
            q_values = policy_net(states.float()).gather(1, actions.long().unsqueeze(1)).squeeze()
            next_q = target_net(next_states.float()).max(1)[0]
            target = rewards.float() + GAMMA * next_q * (1 - dones.float())
            loss = nn.MSELoss()(q_values, target.detach())
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if done:
            break

        # Update target network
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    if episode % 100 == 0:
        print(f"Episode {episode} | Total Reward: {total_reward:.1f}")

# Save models
torch.save(policy_net.state_dict(), 'dqn_policy_net.pth')
print("Training complete.")
