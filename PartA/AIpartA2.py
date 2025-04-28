# part_a.py

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# === Q1: Classification on 10×10 images (two classes only) ===

# Load data
data = np.load("sprites_greyscale10.npz", allow_pickle=True)
train_X = data["train_X"]
train_Y = data["train_Y"]
test_X  = data["test_X"]
test_Y  = data["test_Y"]

# Filter only 'human' and 'orc'
def filter_classes(X, Y, classes=("human","orc")):
    mask = np.isin(Y, classes)
    return X[mask], Y[mask]

train_X_bin, train_Y_bin = filter_classes(train_X, train_Y)
test_X_bin,  test_Y_bin  = filter_classes(test_X,  test_Y)

# Flatten and encode labels
n_train = len(train_X_bin)
n_test  = len(test_X_bin)
train_X_bin = train_X_bin.reshape(n_train, -1)
test_X_bin  = test_X_bin.reshape(n_test,  -1)
le = LabelEncoder()
train_Y_bin = le.fit_transform(train_Y_bin)
test_Y_bin  = le.transform(test_Y_bin)

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(train_X_bin, train_Y_bin)
dt_preds = dt.predict(test_X_bin)
dt_acc   = accuracy_score(test_Y_bin, dt_preds)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_X_bin, train_Y_bin)
knn_preds = knn.predict(test_X_bin)
knn_acc   = accuracy_score(test_Y_bin, knn_preds)

# Train Logistic Regression
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(train_X_bin, train_Y_bin)
lr_preds = lr.predict(test_X_bin)
lr_acc   = accuracy_score(test_Y_bin, lr_preds)

print("\n--- Model Accuracies (human vs orc) ---")
print(f"Decision Tree Accuracy:       {dt_acc:.4f}")
print(f"K-Nearest Neighbors Accuracy: {knn_acc:.4f}")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

# Show some images
fig, axes = plt.subplots(2, 5, figsize=(10,4))
for i, ax in enumerate(axes.flat):
    img = train_X[i].reshape(10,10)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"{train_Y[i]}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# === Q2: Regression on human guards (four features) ===

# Read CSVs
train_data = pd.read_csv("dungeon_sensorstats_train.csv")
test_data  = pd.read_csv("dungeon_sensorstats_test.csv")

# Filter only human
train_h = train_data[train_data['race']=="human"]
test_h  = test_data[test_data['race']=="human"]

# Four-feature model
X_train_4 = train_h[['intelligence','stench','sound','heat']]
y_train_4 = train_h['bribe']
X_test_4  = test_h[['intelligence','stench','sound','heat']]
y_test_4  = test_h['bribe']

model_4 = LinearRegression()
model_4.fit(X_train_4, y_train_4)
y_pred_4 = model_4.predict(X_test_4)
mse_4    = mean_squared_error(y_test_4, y_pred_4)
print(f"\nMSE with 4 features = {mse_4:.4f}")

# Single-feature model (intelligence)
X_train_int = train_h[['intelligence']]
X_test_int  = test_h[['intelligence']]
y_train_int = train_h['bribe']
y_test_int  = test_h['bribe']

model_int = LinearRegression()
model_int.fit(X_train_int, y_train_int)
y_pred_int = model_int.predict(X_test_int)
mse_int    = mean_squared_error(y_test_int, y_pred_int)
print(f"MSE with single feature (intelligence) = {mse_int:.4f}")

plt.figure(figsize=(6,4))
plt.scatter(X_test_int, y_test_int, alpha=0.6, label="Data (test)")
int_range = np.linspace(X_test_int['intelligence'].min(),
                        X_test_int['intelligence'].max(), 100).reshape(-1,1)
plt.plot(int_range, model_int.predict(int_range), linewidth=2, label="Fit")
plt.xlabel("Intelligence")
plt.ylabel("Bribe")
plt.title("Intelligence vs Bribe")
plt.legend()
plt.show()

# === Q3: Clustering on two features (height & weight) ===

# Read full data
full = pd.read_csv("dungeon_sensorstats.csv")
X_cluster = full[['height','weight']].values

# Elbow & silhouette
inertias, sil_scores = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42).fit(X_cluster)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster, km.labels_))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(range(2,11), inertias, 'bx-')
plt.title("Elbow Method"); plt.xlabel("k"); plt.ylabel("Inertia")
plt.subplot(1,2,2)
plt.plot(range(2,11), sil_scores, 'bx-')
plt.title("Silhouette Analysis"); plt.xlabel("k"); plt.ylabel("Score")
plt.tight_layout()
plt.show()

best_k = 3

# KMeans
kmm = KMeans(n_clusters=best_k, random_state=42)
labels_km = kmm.fit_predict(X_cluster)
plt.figure()
plt.scatter(X_cluster[:,0], X_cluster[:,1], c=labels_km, alpha=0.6)
plt.scatter(kmm.cluster_centers_[:,0], kmm.cluster_centers_[:,1],
            marker='X', s=200, c='k', label='Centroids')
plt.xlabel("Height"); plt.ylabel("Weight")
plt.title(f"KMeans (k={best_k})")
plt.legend()
plt.show()

# Gaussian Mixture
gmm = GaussianMixture(n_components=best_k, random_state=42)
labels_gm = gmm.fit_predict(X_cluster)
plt.figure()
plt.scatter(X_cluster[:,0], X_cluster[:,1], c=labels_gm, alpha=0.6)
plt.xlabel("Height"); plt.ylabel("Weight")
plt.title(f"GMM (k={best_k})")
plt.show()
