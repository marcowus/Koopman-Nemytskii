import numpy as np
import matplotlib.pyplot as plt
from systems.tank import TankSystem
from lib.koopman import KoopmanRegressor

# 1. Generate Training Data
# Random step inputs
np.random.seed(42)
t_max = 200.0
dt = 0.5
n_steps = int(t_max / dt)
U_train = np.zeros((n_steps, 1))
for i in range(0, n_steps, 20):
    U_train[i:i+20] = np.random.uniform(0.5, 2.0)

tank = TankSystem(A=1.0, a=0.2, g=9.81)
h0 = 1.0
h_curr = h0
X_train = [h_curr]
for i in range(n_steps):
    h_next = tank.step(h_curr, U_train[i], dt)
    X_train.append(h_next)
    h_curr = h_next

X_all = np.array(X_train)
X_train = X_all[:-1].reshape(-1, 1) # inputs
X_next_train = X_all[1:].reshape(-1, 1) # targets
U_train = U_train.reshape(-1, 1)

# 2. Train Koopman Operator (RRR with RBF Kernel)
# Input features: [h_k, u_k] -> RBF -> High Dim
# Output features: [h_{k+1}] -> RBF -> High Dim (for projection)
# We map input features directly to *lifted output features* via RRR K matrix.
# Then recover h_{k+1} from lifted output features.
koopman = KoopmanRegressor(kernel_type='rbf', rbf_gamma=0.5, rbf_components=200, rank=20, reg_alpha=1e-3)
koopman.fit(X_train, U_train, X_next_train)

# 3. Test on New Inputs (Sine Wave)
t_test = np.arange(0, 100, dt)
U_test = 1.0 + 0.5 * np.sin(0.1 * t_test).reshape(-1, 1)
X_test_true = [h0]
h_curr = h0
for u in U_test:
    h_next = tank.step(h_curr, u, dt)
    X_test_true.append(h_next)
    h_curr = h_next
X_test_true = np.array(X_test_true)

# Predict with Koopman
X_test_pred = [h0]
x_curr = np.array([[h0]])
for u in U_test:
    x_next = koopman.predict(x_curr, u.reshape(1, -1))
    X_test_pred.append(x_next.flat[0])
    x_curr = x_next.reshape(1, -1)

X_test_pred = np.array(X_test_pred)

# 4. Plot
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_test, U_test, 'b-', label='Input u(t)')
plt.ylabel('Flow Rate')
plt.legend()
plt.title('Test Input')

plt.subplot(2, 1, 2)
plt.plot(t_test, X_test_true[:-1], 'k-', label='True h(t)')
plt.plot(t_test, X_test_pred[:-1], 'r--', label='Koopman RRR Prediction')
plt.ylabel('Level h')
plt.legend()
plt.title('Tank Level Prediction')

plt.tight_layout()
plt.savefig('tank_prediction.png')
print("Tank prediction saved to tank_prediction.png")
