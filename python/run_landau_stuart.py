import numpy as np
import matplotlib.pyplot as plt
from systems.landau_stuart import LandauStuart
from lib.koopman import KoopmanRegressor

# 1. Generate Data
dt = 0.1
t_max = 50.0
ls = LandauStuart(mu=0.5, omega=1.0)

# Simulate a trajectory
t_span = (0, t_max)
z0 = [2.0, 0.0]
t, X = ls.simulate(z0, t_span, dt)
# X is (N, 2)

# Prepare Training Data
X_train = X[:-1]
X_next_train = X[1:]
U_train = None # Autonomous system

# 2. Train Koopman Operator
# We use RBF features for stability
koopman = KoopmanRegressor(kernel_type='rbf', rbf_gamma=1.0, rbf_components=100, rank=10, reg_alpha=1e-3)
koopman.fit(X_train, U_train, X_next_train)

# 3. Predict Trajectory
# Start from same initial condition
X_pred = [z0]
x_curr = np.array(z0).reshape(1, -1)

for _ in range(len(t) - 1):
    x_next = koopman.predict(x_curr, U=None)
    X_pred.append(x_next[0])
    x_curr = x_next

X_pred = np.array(X_pred)

# 4. Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(X[:, 0], X[:, 1], 'k-', label='True')
plt.plot(X_pred[:, 0], X_pred[:, 1], 'r--', label='Koopman RRR')
plt.title('Phase Portrait')
plt.legend()
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.plot(t, X[:, 0], 'k-', label='x True')
plt.plot(t, X_pred[:, 0], 'r--', label='x Pred')
plt.title('Time Series')
plt.legend()

plt.tight_layout()
plt.savefig('landau_stuart_prediction.png')
print("Landau-Stuart prediction saved to landau_stuart_prediction.png")
