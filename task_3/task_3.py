import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cpu")

M = 3
input_dim = 2
num_rules = M * M
N = 600


class ANFIS(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.num_rules = M * M

        self.c_x = nn.Parameter(torch.linspace(0.2, 0.8, M).reshape(1, M))
        self.sigma_x = nn.Parameter(torch.ones(1, M) * 0.15)
        self.c_y = nn.Parameter(torch.linspace(0.2, 0.8, M).reshape(1, M))
        self.sigma_y = nn.Parameter(torch.ones(1, M) * 0.15)

        self.next = torch.zeros(self.num_rules, 3, device=device)

    @staticmethod
    def gaussian_mf(x, c, sigma):
        return torch.exp(-0.5 * ((x - c) / (sigma + 1e-6)) ** 2)

    def forward_memberships(self, X):
        x = X[:, 0:1]
        y = X[:, 1:2]
        mu_x = self.gaussian_mf(x, self.c_x, self.sigma_x)
        mu_y = self.gaussian_mf(y, self.c_y, self.sigma_y)
        return mu_x, mu_y

    def forward(self, X):
        mu_x, mu_y = self.forward_memberships(X)
        batch = mu_x.shape[0]
        firing = [(mu_x[:, i] * mu_y[:, j]).reshape(batch, 1)
                  for i in range(self.M) for j in range(self.M)]
        firing = torch.cat(firing, dim=1)
        norm = firing.sum(dim=1, keepdim=True) + 1e-9
        w = firing / norm
        X_ext = torch.cat([X, torch.ones(batch, 1)], dim=1)
        rule_outs = X_ext @ self.next.t()
        y_pred = (w * rule_outs).sum(dim=1, keepdim=True)
        return y_pred, w

    def update_consequents_ls(self, X_np, y_np, reg=1e-6):
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        mu_x, mu_y = self.forward_memberships(X_t)
        batch = X_np.shape[0]
        firing = [(mu_x[:, i] * mu_y[:, j]).reshape(batch, 1)
                  for i in range(self.M) for j in range(self.M)]
        firing = torch.cat(firing, dim=1).cpu().detach().numpy()
        w = firing / (firing.sum(axis=1, keepdims=True) + 1e-9)
        Phi = np.zeros((batch, self.num_rules * 3))
        for r in range(self.num_rules):
            Phi[:, r * 3 + 0] = w[:, r] * X_np[:, 0]
            Phi[:, r * 3 + 1] = w[:, r] * X_np[:, 1]
            Phi[:, r * 3 + 2] = w[:, r]
        A = Phi.T @ Phi + reg * np.eye(Phi.shape[1])
        b = Phi.T @ y_np
        theta = np.linalg.solve(A, b).reshape(self.num_rules, 3)
        self.next = torch.tensor(theta, dtype=torch.float32, device=device)


def target_fn(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


X = np.random.rand(N, 2)
y = target_fn(X[:, 0], X[:, 1]).reshape(-1, 1)

train_idx = np.random.choice(N, int(0.8 * N), replace=False)
test_idx = np.setdiff1d(np.arange(N), train_idx)

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]


model = ANFIS(M)
optimizer = torch.optim.Adam([model.c_x, model.sigma_x, model.c_y, model.sigma_y], lr=0.05)
epochs = 30
train_losses, test_losses = [], []

for ep in range(epochs):
    model.update_consequents_ls(X_train, y_train)
    model.eval()
    with torch.no_grad():
        y_pred_train, _ = model.forward(torch.tensor(X_train, dtype=torch.float32))
        y_pred_test, _ = model.forward(torch.tensor(X_test, dtype=torch.float32))
        train_mse = ((y_pred_train.cpu().numpy() - y_train) ** 2).mean()
        test_mse = ((y_pred_test.cpu().numpy() - y_test) ** 2).mean()
    train_losses.append(train_mse)
    test_losses.append(test_mse)

    model.train()
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    y_pred, _ = model.forward(X_t)
    loss = nn.MSELoss()(y_pred, y_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (ep + 1) % 5 == 0 or ep == 0:
        print(f"Epoch {ep+1}/{epochs}  Train MSE={train_mse:.6f}  Test MSE={test_mse:.6f}")


with torch.no_grad():
    y_pred_all, _ = model.forward(torch.tensor(X, dtype=torch.float32))
    y_pred_all = y_pred_all.cpu().numpy()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.title("MSE over epochs")

plt.subplot(1, 2, 2)
plt.scatter(y, y_pred_all, s=10)
plt.plot([-1, 1], [-1, 1], 'r')
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("True vs Predicted")
plt.tight_layout()
plt.show()
