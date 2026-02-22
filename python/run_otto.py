import numpy as np
import matplotlib.pyplot as plt
from systems.otto import OttoCycle
from lib.koopman import KoopmanRegressor

# 1. Generate Training Data
# Random inputs: Throttle (theta), Load Torque (T_load)
t_max = 5.0 # Seconds (engine is fast)
dt = 0.01
n_steps = int(t_max / dt)

# Inputs
theta_train = np.linspace(0.2, 0.8, n_steps).reshape(-1, 1) # Ramp throttle
load_train = 5.0 * np.ones((n_steps, 1)) # Constant load
U_train = np.hstack([theta_train, load_train])

otto = OttoCycle()
x0 = [101325.0, 100.0] # Initial Pm, w (rad/s)
X_train = [x0]
x_curr = x0

for i in range(n_steps):
    x_next = otto.step(x_curr, U_train[i], dt)
    X_train.append(x_next)
    x_curr = x_next

X_all = np.array(X_train)
X_train = X_all[:-1]
X_next_train = X_all[1:]

# 2. Train Koopman Operator (RRR with RBF Kernel)
# Input features: [Pm, w, theta, T_load] -> RBF -> High Dim
# Output features: [Pm_next, w_next] -> Lifted -> Projected
koopman = KoopmanRegressor(kernel_type='rbf', rbf_gamma=0.001, rbf_components=300, rank=50, reg_alpha=1e-3)
koopman.fit(X_train, U_train, X_next_train)

# 3. Test on Step Input
t_test = np.arange(0, 2.0, dt)
n_test = len(t_test)
U_test = np.zeros((n_test, 2))
U_test[:, 0] = 0.5 # Constant throttle
U_test[:, 1] = 5.0 # Constant load

X_test_true = [x0]
x_curr = x0
for i in range(n_test):
    x_next = otto.step(x_curr, U_test[i], dt)
    X_test_true.append(x_next)
    x_curr = x_next
X_test_true = np.array(X_test_true)

# Predict with Koopman
X_test_pred = [x0]
x_curr = np.array([x0])
for i in range(n_test):
    x_next = koopman.predict(x_curr, U_test[i].reshape(1, -1))
    X_test_pred.append(x_next[0])
    x_curr = x_next
X_test_pred = np.array(X_test_pred)

# 4. Plot
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t_test, X_test_true[:-1, 1], 'k-', label='True Speed (w)')
plt.plot(t_test, X_test_pred[:-1, 1], 'r--', label='Koopman Pred')
plt.ylabel('Engine Speed (rad/s)')
plt.legend()
plt.title('Otto Cycle Engine Speed Prediction')

plt.subplot(2, 1, 2)
plt.plot(t_test, X_test_true[:-1, 0], 'k-', label='True Manifold Pressure (Pm)')
plt.plot(t_test, X_test_pred[:-1, 0], 'r--', label='Koopman Pred')
plt.ylabel('Pressure (Pa)')
plt.legend()

plt.tight_layout()
plt.savefig('otto_prediction.png')
print("Otto prediction saved to otto_prediction.png")
