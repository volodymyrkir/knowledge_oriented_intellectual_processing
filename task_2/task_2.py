import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import skfuzzy as fuzz
from scipy.linalg import lstsq
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

dataset = fetch_california_housing()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def build_mlp(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

mlp_model = build_mlp(X_train_scaled.shape[1])
mlp_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

y_pred_mlp = mlp_model.predict(X_test_scaled).flatten()
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)


num_rules = 5
m = 2.0

cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_train_scaled.T, num_rules, m, error=0.005, maxiter=1000, init=None)

sigmas = np.zeros((num_rules, X_train_scaled.shape[1]))
for k in range(num_rules):
    for j in range(X_train_scaled.shape[1]):
        weighted_var = np.sum(u[k] ** m * (X_train_scaled[:, j] - cntr[k, j]) ** 2) / np.sum(u[k] ** m)
        sigmas[k, j] = np.sqrt(weighted_var) if weighted_var > 0 else 1.0

def compute_firing_strengths(X, cntr, sigmas):
    n_samples, n_features = X.shape
    w = np.zeros((n_samples, num_rules))
    for i in range(n_samples):
        for k in range(num_rules):
            mf_values = np.exp(-0.5 * ((X[i] - cntr[k]) ** 2 / (sigmas[k] ** 2 + 1e-6)))
            w[i, k] = np.prod(mf_values)
    return w

w_train = compute_firing_strengths(X_train_scaled, cntr, sigmas)
beta_train = w_train / (np.sum(w_train, axis=1, keepdims=True) + 1e-6)  # Normalize

n_features = X_train_scaled.shape[1]
Phi = np.zeros((X_train_scaled.shape[0], num_rules * (n_features + 1)))
for k in range(num_rules):
    start = k * (n_features + 1)
    Phi[:, start] = beta_train[:, k]  # p_k0 term
    Phi[:, start + 1 : start + 1 + n_features] = beta_train[:, k, None] * X_train_scaled

p, _, _, _ = lstsq(Phi, y_train)

def tsk_predict(X, cntr, sigmas, p):
    w = compute_firing_strengths(X, cntr, sigmas)
    beta = w / (np.sum(w, axis=1, keepdims=True) + 1e-6)
    y_pred = np.zeros(X.shape[0])
    n_features = X.shape[1]
    for k in range(num_rules):
        start = k * (n_features + 1)
        y_k = p[start] + np.dot(X, p[start + 1 : start + 1 + n_features])
        y_pred += beta[:, k] * y_k
    return y_pred

y_pred_tsk = tsk_predict(X_test_scaled, cntr, sigmas, p)
mse_tsk = mean_squared_error(y_test, y_pred_tsk)
r2_tsk = r2_score(y_test, y_pred_tsk)


print("Performance Comparison:")
print(f"MLP - MSE: {mse_mlp:.4f}, R²: {r2_mlp:.4f}")
print(f"TSK Fuzzy - MSE: {mse_tsk:.4f}, R²: {r2_tsk:.4f}")
if mse_mlp < mse_tsk:
    print("MLP outperforms TSK Fuzzy in terms of MSE.")
else:
    print("TSK Fuzzy outperforms MLP in terms of MSE.")
