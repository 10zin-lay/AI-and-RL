import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# === Load Data ===
data = np.load("sprites_greyscale10.npz", allow_pickle=True)
train_X = data["train_X"]
train_Y = data["train_Y"]
test_X = data["test_X"]
test_Y = data["test_Y"]

# === Confirm Class Labels ===
print("Train labels:", np.unique(train_Y))
print("Test labels:", np.unique(test_Y))

# === Filter Only 'human' and 'orc' Classes ===
def filter_classes(X, Y, class_names=("human", "orc")):
    mask = np.isin(Y, class_names)
    return X[mask], Y[mask]

train_X_bin, train_Y_bin = filter_classes(train_X, train_Y, ("human", "orc"))
test_X_bin, test_Y_bin = filter_classes(test_X, test_Y, ("human", "orc"))

# === Flatten Images ===
train_X_bin = train_X_bin.reshape(len(train_X_bin), -1)
test_X_bin = test_X_bin.reshape(len(test_X_bin), -1)

# === Convert String Labels to Integers ===
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_Y_bin = le.fit_transform(train_Y_bin)
test_Y_bin = le.transform(test_Y_bin)

# === Train Classifiers ===

# Decision Tree
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(train_X_bin, train_Y_bin)
dt_preds = dt.predict(test_X_bin)
dt_acc = accuracy_score(test_Y_bin, dt_preds)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_X_bin, train_Y_bin)
knn_preds = knn.predict(test_X_bin)
knn_acc = accuracy_score(test_Y_bin, knn_preds)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(train_X_bin, train_Y_bin)
lr_preds = lr.predict(test_X_bin)
lr_acc = accuracy_score(test_Y_bin, lr_preds)

# === Output Results ===
print("\n--- Model Accuracies ---")
print(f"Decision Tree Accuracy:       {dt_acc:.4f}")
print(f"K-Nearest Neighbors Accuracy: {knn_acc:.4f}")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
# Assuming you've already loaded this:
# train_X, train_Y = data["train_X"], data["train_Y"]
# and it's currently shape (N, 100)

# Show first 10 images (reshaped to 10x10)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    image = train_X[i].reshape(10, 10)  # 🔥 FIX: reshape to 10x10
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {train_Y[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()
target_class = "human"
indices = np.where(train_Y == target_class)[0]
sample_indices = random.sample(list(indices), 5)

for idx in sample_indices:
    image = train_X[idx].reshape(10, 10)  # 🔥 FIX: reshape to 10x10
    plt.imshow(image, cmap="gray")
    plt.title(f"Label: {train_Y[idx]}")
    plt.axis("off")
    plt.show()
best_depth = 1
best_acc = 0
print("Tuning Decision Tree:")

for depth in range(1, 20):
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(train_X_bin, train_Y_bin)
    preds = model.predict(test_X_bin)
    acc = accuracy_score(test_Y_bin, preds)
    print(f"  Depth {depth}: Accuracy = {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_depth = depth

print(f"\nBest Decision Tree Depth = {best_depth} with Accuracy = {best_acc:.4f}")
best_k = 1
best_acc = 0
print("\nTuning KNN:")

for k in range(1, 20):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_X_bin, train_Y_bin)
    preds = model.predict(test_X_bin)
    acc = accuracy_score(test_Y_bin, preds)
    print(f"  k = {k}: Accuracy = {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"\nBest k = {best_k} with Accuracy = {best_acc:.4f}")
best_c = 0.01
best_acc = 0
print("\nTuning Logistic Regression (C):")

for c in [0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(C=c, max_iter=5000)
    model.fit(train_X_bin, train_Y_bin)
    preds = model.predict(test_X_bin)
    acc = accuracy_score(test_Y_bin, preds)
    print(f"  C = {c}: Accuracy = {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_c = c

print(f"\nBest C = {best_c} with Accuracy = {best_acc:.4f}")
# Use your trained model — choose one
model = lr  # or dt or lr

# Number of samples to show
n_samples = 5

# Pick random test indices
random_indices = random.sample(range(len(test_X_bin)), n_samples)

# Predict on the test set
predictions = model.predict(test_X_bin)

correct = 0

# Plot each sample
for i, idx in enumerate(random_indices):
    img = test_X_bin[idx].reshape(10, 10)  # reshape for display
    actual_label = le.inverse_transform([test_Y_bin[idx]])[0]
    predicted_label = le.inverse_transform([predictions[idx]])[0]

    # Accuracy count
    if actual_label == predicted_label:
        correct += 1

    # Show image
    plt.imshow(img, cmap="gray")
    plt.title(f"Actual: {actual_label} | Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()

# Print accuracy on entire test set
from sklearn.metrics import accuracy_score
acc = accuracy_score(test_Y_bin, predictions)
print(f"\n✅ Overall Test Accuracy: {acc:.4f}")
# (2.1) Read the CSV files
train_data = pd.read_csv("dungeon_sensorstats_train.csv")
test_data  = pd.read_csv("dungeon_sensorstats_test.csv")

# (2.2) Extract the subset of the training and test data
#       - Only keep "human" race
#       - Only keep the four sensor features and 'bribe'
# -----------------------------------------------------
train_data_human = train_data[train_data['race'] == 'human'].copy()
test_data_human  = test_data[test_data['race'] == 'human'].copy()

# Extract features and target for training
X_train_4 = train_data_human[['intelligence', 'stench', 'sound', 'heat']]
y_train_4 = train_data_human['bribe']

# Extract features and target for testing
X_test_4 = test_data_human[['intelligence', 'stench', 'sound', 'heat']]
y_test_4 = test_data_human['bribe']

# (2.3) Build a Linear Regression model using the 4 features
#       and output the MSE
# ----------------------------------------------------------
model_4 = LinearRegression()
model_4.fit(X_train_4, y_train_4)

# Predict on the test set
y_pred_4 = model_4.predict(X_test_4)

# Compute the Mean Squared Error (MSE)
mse_4 = mean_squared_error(y_test_4, y_pred_4)
print("MSE with 4 features =", mse_4)


# (2.4) Single-feature Linear Regression
#       Choose one feature, e.g. "intelligence"
#       Build a model, output MSE, and plot the regression.
# ---------------------------------------------------------
X_train_int = train_data_human[['intelligence']]
X_test_int  = test_data_human[['intelligence']]
y_train_int = train_data_human['bribe']
y_test_int  = test_data_human['bribe']

model_int = LinearRegression()
model_int.fit(X_train_int, y_train_int)

# Predict on the test set
y_pred_int = model_int.predict(X_test_int)
mse_int = mean_squared_error(y_test_int, y_pred_int)
print("MSE with single feature (intelligence) =", mse_int)

# Plot the (intelligence vs. bribe) data + model fit
plt.figure(figsize=(6,4))
# Scatter of test set
plt.scatter(X_test_int, y_test_int, color='blue', alpha=0.6, label='Data (test set)')

# Generate a smooth range of intelligence values for plotting the regression line
int_range = np.linspace(X_test_int['intelligence'].min(),
                        X_test_int['intelligence'].max(), 100).reshape(-1, 1)

# Predict bribe values for the smooth range
bribe_pred_line = model_int.predict(int_range)
plt.plot(int_range, bribe_pred_line, color='red', linewidth=2, label='Regression line')

plt.xlabel("Intelligence")
plt.ylabel("Bribe")
plt.title("Single-Feature Linear Regression (Intelligence vs. Bribe)")
plt.legend()
plt.show()
train_data = pd.read_csv("dungeon_sensorstats.csv")
Train_data= train_data[['height','weight','strength']]
X=Train_data
kmm= KMeans(n_clusters=5, random_state=42)
kmm.fit(X)
fig, axs = plt.subplots(3,3, figsize=(10,10), sharey='row', sharex='col')

# Convert the DataFrame to a NumPy array for slicing
X_np = X.to_numpy()

for i in range(3):  # Change loop range to 3 instead of 4
    for j in range(3):
        # Use the NumPy array for scatter plot
        axs[j,i].scatter(X_np[:,i],X_np[:,j], c=kmm.labels_)
        if j == 2:  # Change condition to j == 2 for x-axis labels
            axs[j,i].set_xlabel(X.columns[i])  # Use column names for labels
        if i == 0:
            axs[j,i].set_ylabel(X.columns[j])  # Use column names for labels

plt.show() # Show the plot
inertias = []
K = 10
for k in range(1, K+1):
    kmm = KMeans(n_clusters=k).fit(X)
    kmm.fit(X)
    inertias.append(kmm.inertia_)
# Plot the elbow
plt.figure();
plt.plot(range(1, K+1), inertias, 'bx-');
plt.xlabel('k');
plt.ylabel('Inertia');
plt.title('The elbow method showing the optimal k');
# Set up a variable to store the silhoutte scores
silhouette_scores = []
K = 10
# Loop over values of k from 2 to 10 for k in range(2, K+1):
for k in range(2, K+1):  # Instantiate the KMeans class with k clusters
 kmm = KMeans(n_clusters=k, random_state=42)
# Fit the model to the data
 kmm.fit(X)
# Store the value of the silhouette score for this value of k silhouette_scores.append(silhouette_score(X, kmm.labels_))
 silhouette_scores.append(silhouette_score(X, kmm.labels_))
# Plot the scores
plt.figure()
plt.plot(range(2, K+1), silhouette_scores, 'bx-');
plt.xlabel('k');
plt.ylabel('Silhouette Score');
plt.title('Silhouette scores showing optimal k');