import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd


# Input data, formula = x^2 + 2sin(1.5x) + 2 * NOISE
N = 300
x = np.linspace(-5, 5, N).reshape(-1, 1)    # форма (N, d) ; d=1
y = (x[:, 0] ** 2) + 2.0 * np.sin(1.5 * x[:, 0]) + 2.0 * np.random.randn(N) * 0.5
y = y.reshape(-1, 1)

df = pd.DataFrame(np.hstack([x, y]), columns=['x', 'y'])


m = 2
data_for_cmeans = x.T
cntr, u, u0, dists, jm, p, fpc = fuzz.cluster.cmeans(
    data_for_cmeans, c=m, m=2.0, error=1e-6, maxiter=1000, init=None
)

centers = cntr  # (m, d)
print("Centers (rules):\n", centers)


d = x.shape[1]


sigmas = np.zeros((m, d))
for i in range(m):
    for dim in range(d):
        # відстані до інших центрів по виміру dim
        dist = np.abs(centers[:, dim] - centers[i, dim])
        # уникаємо нуля (сам до себе) - візьмемо середню по іншим
        if m > 1:
            sig = np.mean(dist[dist > 1e-8])
            if np.isnan(sig) or sig <= 0:
                sig = 1.0
        else:
            sig = 1.0
        sigmas[i, dim] = sig

sigmas[sigmas <= 1e-6] = 1.0


def gauss_mf(x_input, c, sigma):
    return np.exp(-0.5 * ((x_input - c) / sigma) ** 2)


def compute_firing_strengths(X):
    N = X.shape[0]
    W = np.ones((N, m))
    for i in range(m):
        w_i = np.ones(N)
        for dim in range(d):
            c = centers[i, dim]
            sigma = sigmas[i, dim]
            w_dim = gauss_mf(X[:, dim], c, sigma)
            w_i = w_i * w_dim
        W[:, i] = w_i
    return W


W = compute_firing_strengths(x)
# перевіримо
print("Firing strengths (пʼять перших зразків):\n", W[:5, :])


N = x.shape[0]
Z = np.zeros((N, m * (d + 1)))
for i in range(m):
    wi = W[:, i].reshape(-1, 1)
    Xi_aug = np.hstack([x, np.ones((N, 1))])
    Z[:, i * (d + 1):(i + 1) * (d + 1)] = wi * Xi_aug

lambda_reg = 1e-3
A = Z.T.dot(Z) + lambda_reg * np.eye(Z.shape[1])
b_vec = Z.T.dot(y).reshape(-1, )
Theta = np.linalg.solve(A, b_vec)
consequents = Theta.reshape(m, d + 1)  # кожний рядок: [a_i1 ... a_id, b_i]
print("Consequent params (a's and b's) per rule:\n", consequents)

def predict(X):
    N = X.shape[0]
    W = compute_firing_strengths(X)
    w_sum = np.sum(W, axis=1).reshape(-1, 1)
    w_sum[w_sum == 0] = 1e-8

    Xi_aug = np.hstack([X, np.ones((N, 1))])
    Fi = np.zeros((N, m))
    for i in range(m):
        a_b = consequents[i]
        Fi[:, i] = Xi_aug.dot(a_b)

    W_norm = W / w_sum
    y_hat = np.sum(W_norm * Fi, axis=1)
    return y_hat.reshape(-1, 1)

y_hat = predict(x)

mse = np.mean((y_hat - y) ** 2)
print(f"Train MSE: {mse:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(x[:, 0], y[:, 0], s=10, label='True data', alpha=0.5)
plt.scatter(x[:, 0], y_hat[:, 0], s=10, label='TSK prediction', alpha=0.8)
plt.title(f"TSK model (m={m}) — Train MSE={mse:.3f}")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 3))
plt.scatter(centers[:, 0], np.zeros_like(centers[:, 0]), c='red', label='rule centers')
for i in range(m):
    c = centers[i, 0]
    s = sigmas[i, 0]
    xs = np.linspace(c - 3*s, c + 3*s, 200)
    plt.plot(xs, gauss_mf(xs, c, s), label=f'rule {i}')
plt.title("Membership functions (1D projections)")
plt.legend()
plt.show()
