import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

n = 25
x1 = np.array([0, 0, 0, 1, 1, 2, 2, 2])
x2 = np.array([1.5, 2.5, 3.5, 1.5, 3.5, 1.5, 2.5, 2.5])
y = np.array([
    2.3,
    4 + 0.3 * n,
    2 - 0.1 * n,
    5 - 0.2 * n,
    4 - 0.2 * n,
    6.1 + 0.2 * n,
    6.5 - 0.1 * n,
    7.2
])

X_mat = np.column_stack((np.ones(len(x1)), x1, x2))
coeffs = np.linalg.lstsq(X_mat, y, rcond=None)[0]
a0, a1, a2 = coeffs

y_pred = X_mat @ coeffs
r2 = r2_score(y, y_pred)
y_mean = np.mean(y)

x1_t, x2_t = 1.5, 3.0
y_val = a0 + a1 * x1_t + a2 * x2_t

print(f"Sums: x1={np.sum(x1)}, x2={np.sum(x2)}, y={np.sum(y):.2f}")
print(f"Sums squares: x1^2={np.sum(x1**2)}, x2^2={np.sum(x2**2)}, x1x2={np.sum(x1*x2)}")
print(f"Sums products: x1y={np.sum(x1*y):.2f}, x2y={np.sum(x2*y):.2f}")
print(f"Coefficients: a0={a0:.4f}, a1={a1:.4f}, a2={a2:.4f}")
print(f"Equation: y = {a0:.4f} + {a1:.4f}*x1 + ({a2:.4f})*x2")
print(f"Value at (1.5, 3): {y_val:.4f}")
print(f"R^2: {r2:.4f}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, color='red', s=100, label='Data points')

x1_surf, x2_surf = np.meshgrid(np.linspace(0, 2, 20), np.linspace(1.5, 3.5, 20))
y_surf = a0 + a1 * x1_surf + a2 * x2_surf

ax.plot_surface(x1_surf, x2_surf, y_surf, alpha=0.4, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.legend()
plt.show()