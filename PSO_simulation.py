import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def schwefel(x1, x2):
    return 418.9829 * 2 - x1 * np.sin(np.sqrt(abs(x1))) - x2 * np.sin(np.sqrt(abs(x2)))

def banana(x1, x2):
    a = 1. - x1
    b = x2 - x1 * x1
    return np.clip(a * a + b * b * 100., 0, 1000)

# Constants for the PSO algorithm
COGNITIVE = 0.1
SOCIAL = 0.4
INERTIA_WEIGHT = 0.7

# SCHWEFEL FUNCTION

# Define the function's domain and the true global minimum
x = np.linspace(-500, 500, 1000)
y = np.linspace(-500, 500, 1000)
X1, X2 = np.meshgrid(x, y)
Z = schwefel(X1, X2)
x_min = 420.9687
y_min = 420.9687

# Initialize random positions and velocities for particles.
n_particles = 50
np.random.seed(100)
X = np.random.uniform(-500, 500, (2, n_particles))
V = np.random.randn(2, n_particles) * 0.3

# Initialize data for the algorithm.
pbest = X
pbest_cost = schwefel(X[0], X[1])
gbest = pbest[:, pbest_cost.argmin()]
gbest_cost = pbest_cost.min()

# update function is ran on each pso iteration.
def update(cost_func):
    global V, X, pbest, pbest_cost, gbest, gbest_cost
    r1, r2 = np.random.random(2)
    # update particle velocities.
    V = INERTIA_WEIGHT * V + COGNITIVE * r1 *(pbest - X) + SOCIAL * r2 * (gbest.reshape(-1, 1) - X)
    X = X + V
    cost = cost_func(X[0], X[1])
    pbest[:, (pbest_cost >= cost)] = X[:, (pbest_cost >= cost)]
    pbest_cost = np.array([pbest_cost, cost]).min(axis=0)
    gbest = pbest[:, pbest_cost.argmin()]
    gbest_cost = pbest_cost.min()

def animate(i, ax, cost_func):
    title = 'Iter: {:02d}'.format(i)
    update(cost_func)
    ax.set_title(title)
    pbest_plot.set_offsets(pbest.T)
    p_plot.set_offsets(X.T)
    gbest_plot.set_offsets(gbest.reshape(1,-1))
    return ax, pbest_plot, p_plot, gbest_plot

fig, ax = plt.subplots(figsize=(10,8))
fig.set_tight_layout(True)
im = ax.imshow(Z, extent=[-500, 501, -500, 501], origin='lower', cmap='RdBu', alpha=0.4)
pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
p_plot = ax.scatter(X[0], X[1], marker='o', color='lime', alpha=0.8)
gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.5)
cset = plt.contour(X1, X2, Z, cmap="RdBu")
plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
plt.colorbar(im)

# Create animation for Schwefel function
anim = FuncAnimation(fig, animate, fargs=(ax, schwefel), frames=list(range(1,100)), interval=150, blit=False, repeat=True) 
anim.save("PSO_schwefel.gif", dpi=120)

print("PSO found best solution at schwefel({})={}".format(gbest, gbest_cost))
print("Global minimum at schwefel({})={}".format([x_min,y_min], schwefel(x_min,y_min)))


# BANANA FUNCTION

# Define the function's domain and the true global minimum for Banana function
x = np.linspace(-2, 2, 1000)
y = np.linspace(-1, 3, 1000)
X1, X2 = np.meshgrid(x, y)
Z = banana(X1, X2)
x_min = 1
y_min = 1

# Initialize random positions and velocities for particles.
X = np.random.uniform(-2, 2, (2, n_particles))
V = np.random.randn(2, n_particles) * 0.3

# Initialize data for the algorithm.
pbest = X
pbest_cost = banana(X[0], X[1])
gbest = pbest[:, pbest_cost.argmin()]
gbest_cost = pbest_cost.min()

# Create new figure and axes

fig2, ax2 = plt.subplots(figsize=(10,8))  # Create new figure and axes
fig2.set_tight_layout(True)
im = ax2.imshow(Z, extent=[-2, 3, -2, 3], origin='lower', cmap='RdBu', alpha=0.4)
pbest_plot = ax2.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
p_plot = ax2.scatter(X[0], X[1], marker='o', color='lime', alpha=0.8)
gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.5)
cset = plt.contour(X1, X2, Z, cmap="RdBu")
plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
plt.colorbar(im)

# Create animation for Banana function
anim = FuncAnimation(fig2, animate, fargs=(ax2, banana), frames=list(range(1,100)), interval=150, blit=False, repeat=True) 
anim.save("PSO_banana.gif", dpi=120)

print("PSO found best solution at banana({})={}".format(gbest, gbest_cost))
print("Global minimum at banana({})={}".format([x_min,y_min], banana(x_min,y_min)))