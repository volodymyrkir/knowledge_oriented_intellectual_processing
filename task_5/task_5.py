import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PSO:
    def __init__(self, func, dim, n_particles=30, iters=100,
                 w=0.7, c1=1.5, c2=1.5, bounds=None):
        self.func = func
        self.dim = dim
        self.n_particles = n_particles
        self.iters = iters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds if bounds is not None else [(-5.12, 5.12)] * dim

        self.pos = np.array([
            np.random.uniform(low=b[0], high=b[1], size=n_particles)
            for b in self.bounds
        ]).T

        self.vel = np.zeros_like(self.pos)

        self.pbest_pos = self.pos.copy()
        self.pbest_cost = np.array([self.func(p) for p in self.pos])

        best_idx = np.argmin(self.pbest_cost)
        self.gbest_pos = self.pbest_pos[best_idx].copy()
        self.gbest_cost = self.pbest_cost[best_idx]

        self.gbest_history = [self.gbest_cost]
        self.trajectories = [self.pos.copy()]

    def step(self):
        r1 = np.random.rand(self.n_particles, self.dim)
        r2 = np.random.rand(self.n_particles, self.dim)

        cognitive = self.c1 * r1 * (self.pbest_pos - self.pos)
        social = self.c2 * r2 * (self.gbest_pos - self.pos)

        self.vel = self.w * self.vel + cognitive + social
        self.pos += self.vel

        for d in range(self.dim):
            low, high = self.bounds[d]
            self.pos[:, d] = np.clip(self.pos[:, d], low, high)

        costs = np.array([self.func(p) for p in self.pos])

        mask = costs < self.pbest_cost
        self.pbest_cost[mask] = costs[mask]
        self.pbest_pos[mask] = self.pos[mask]


        best_idx = np.argmin(self.pbest_cost)
        if self.pbest_cost[best_idx] < self.gbest_cost:
            self.gbest_cost = self.pbest_cost[best_idx]
            self.gbest_pos = self.pbest_pos[best_idx].copy()

        self.gbest_history.append(self.gbest_cost)
        self.trajectories.append(self.pos.copy())

    def optimize(self):
        for _ in range(self.iters):
            self.step()
        return self.gbest_pos, self.gbest_cost


# Тестова функція Растригина
def rastrigin(x):
    A = 10
    x = np.asarray(x)
    return A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x))


# Параметри PSO
dim = 2
pso = PSO(
    func=rastrigin,
    dim=dim,
    n_particles=40,
    iters=120,
    w=0.72,
    c1=1.5,
    c2=1.5,
    bounds=[(-5.12, 5.12)] * dim
)

best_pos, best_cost = pso.optimize()

print("Best position:", best_pos)
print("Best cost:", best_cost)


# Графік збіжності
plt.figure(figsize=(8, 4))
plt.plot(pso.gbest_history, linewidth=2)
plt.title("Convergence of PSO")
plt.xlabel("Iteration")
plt.ylabel("Best cost")
plt.grid(True)
plt.show()


# Контурна карта функції + траєкторії частинок
x = np.linspace(-5.12, 5.12, 200)
y = np.linspace(-5.12, 5.12, 200)
X, Y = np.meshgrid(x, y)
Z = np.array([rastrigin([xx, yy]) for xx, yy in zip(X.ravel(), Y.ravel())])
Z = Z.reshape(X.shape)

plt.figure(figsize=(7, 6))
plt.contourf(X, Y, Z, levels=50, cmap="viridis")

num_to_plot = min(10, pso.n_particles)
for i in range(num_to_plot):
    traj = np.array([pos[i] for pos in pso.trajectories])
    plt.plot(traj[:, 0], traj[:, 1])
    plt.scatter(traj[0, 0], traj[0, 1], marker='o')
    plt.scatter(traj[-1, 0], traj[-1, 1], marker='x')

plt.title("Particle Trajectories over Rastrigin Function")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# Друк фінальних результатів
final_positions = pso.trajectories[-1]
final_costs = np.array([rastrigin(p) for p in final_positions])

df = pd.DataFrame({
    "x1": final_positions[:, 0],
    "x2": final_positions[:, 1],
    "cost": final_costs
}).sort_values("cost")

print(df.head(10))
