# 项目原理与数学分析：非线性控制的Koopman算子方法

**注意**：由于当前代码库中的所有源文件（`.m` 和 `.mat`）内容为空，以下分析基于文件命名结构（如 `landau_stuart.m`, `model1_learning_DMD.mat`, `kernel_input`, `RRR` 等）以及该领域的标准文献和方法推断得出。该项目旨在利用Koopman算子理论，结合核方法（Kernel Methods）和降秩回归（Reduced Rank Regression, RRR），实现非线性系统的辨识、预测与控制。

---

## 1. 引言 (Introduction)

本项目旨在解决非线性动力系统的建模与控制问题。传统的非线性控制方法（如反馈线性化、滑模控制）通常依赖于精确的物理模型，且设计复杂。Koopman算子理论提供了一种全新的视角：通过将非线性系统的状态提升到一个无穷维的希尔伯特空间（Hilbert Space），使得在该空间内的演化呈现为线性的。这使得我们可以利用成熟的线性控制理论（如LQR, MPC）来处理非线性问题。

项目涵盖了三种典型的非线性系统：
1.  **Landau-Stuart 方程**：描述极限环振荡的标准模型。
2.  **水箱系统 (Tank System)**：流体过程控制中的经典非线性模型。
3.  **Otto 循环 (Otto Cycle)**：热力学发动机模型，具有强非线性。

---

## 2. 数学基础 (Mathematical Foundations)

### 2.1 Koopman 算子理论 (Koopman Operator Theory)

考虑一个离散时间的非线性动力系统：
$$ x_{k+1} = f(x_k) $$
其中 $x_k \in \mathcal{M} \subseteq \mathbb{R}^n$ 是状态向量，$f: \mathcal{M} \to \mathcal{M}$ 是演化映射。

Koopman算子 $\mathcal{K}$ 是一个定义在观测函数空间 $\mathcal{F}$（例如 $L^2(\mathcal{M})$）上的无穷维线性算子。对于任意观测函数 $g: \mathcal{M} \to \mathbb{C}$，Koopman算子定义为：
$$ \mathcal{K}g(x_k) = g(f(x_k)) = g(x_{k+1}) $$
这意味着，Koopman算子将观测函数 $g$ 沿系统轨迹向前推演一步。尽管原系统 $f$ 是非线性的，但算子 $\mathcal{K}$ 是线性的：
$$ \mathcal{K}(\alpha g_1 + \beta g_2) = \alpha \mathcal{K}g_1 + \beta \mathcal{K}g_2 $$

### 2.2 扩展动态模态分解 (Extended Dynamic Mode Decomposition, EDMD)

为了在计算机上近似无穷维的Koopman算子，我们使用EDMD算法。选择一组有限的字典函数（observables） $\Psi(x) = [\psi_1(x), \psi_2(x), \dots, \psi_N(x)]^T$。

假设Koopman算子在该子空间内可以近似为一个矩阵 $K \in \mathbb{R}^{N \times N}$：
$$ \Psi(x_{k+1}) \approx K \Psi(x_k) $$

给定数据集 $\{ (x_k, y_k) \}_{k=1}^M$，其中 $y_k = x_{k+1}$。构建数据矩阵：
$$ \Psi_X = [\Psi(x_1), \dots, \Psi(x_M)], \quad \Psi_Y = [\Psi(y_1), \dots, \Psi(y_M)] $$

我们的目标是找到矩阵 $K$，使得预测误差最小：
$$ \min_K \| \Psi_Y - K \Psi_X \|_F^2 $$
其最小二乘解为：
$$ K = \Psi_Y \Psi_X^\dagger $$
其中 $\Psi_X^\dagger$ 是 $\Psi_X$ 的伪逆。

### 2.3 核方法 (Kernel Methods / KDMD)

当字典函数维度 $N$ 很高甚至无穷大时（例如使用高斯核），直接计算 $\Psi_X$ 是不可行的。利用核技巧（Kernel Trick），我们可以避免显式计算特征映射。

定义核函数 $k(x, x') = \langle \Psi(x), \Psi(x') \rangle$。
常见的核函数包括：
-   **高斯径向基函数 (RBF)**: $k(x, x') = \exp(-\frac{\|x-x'\|^2}{2\sigma^2})$
-   **多项式核**: $k(x, x') = (1 + x^T x')^d$

在KDMD中，我们可以通过核矩阵 $G_{ij} = k(x_i, x_j)$ 来隐式地求解Koopman算子的特征值和特征模态，从而实现对非线性系统的特征提取。项目中的 `model1_kernel_input.m` 和 `model1_kernel_state.m` 即使在处理输入和状态时使用了核方法。

### 2.4 降秩回归 (Reduced Rank Regression, RRR)

为了提高模型的泛化能力并提取主要动力学特征，项目中引入了降秩回归（参考 `model1_learning_RRR.mat`）。

RRR 的目标是在约束系数矩阵 $K$ 的秩的情况下最小化误差：
$$ \min_{K} \| Y - X K \|_F^2 \quad \text{s.t.} \quad \text{rank}(K) \le r $$
其中 $r < N$。

**定理 (RRR解)**：
令 $\hat{K}_{OLS} = (X^T X)^{-1} X^T Y$ 为普通最小二乘解。RRR的解可以通过对加权协方差矩阵进行奇异值分解（SVD）得到。具体地，令 $V$ 为 $Y^T X (X^T X)^{-1} X^T Y$ 的前 $r$ 个特征向量组成的矩阵，则：
$$ \hat{K}_{RRR} = \hat{K}_{OLS} V V^T $$
这相当于将最小二乘解投影到由主要动力学模态张成的低维子空间上。

---

## 3. 具体系统模型 (Specific System Models)

### 3.1 Landau-Stuart 方程 (`landau_stuart.m`)

Landau-Stuart 方程是描述非线性振荡和极限环分岔的经典模型。其复数形式为：
$$ \dot{z} = (\mu + i\omega - |z|^2) z $$
其中 $z \in \mathbb{C}$，$\mu$ 是分岔参数，$\omega$ 是角频率。

在极坐标 $z = r e^{i\theta}$ 下：
$$ \dot{r} = \mu r - r^3 $$
$$ \dot{\theta} = \omega $$

当 $\mu > 0$ 时，系统存在一个稳定的极限环 $r = \sqrt{\mu}$。该系统常用于测试Koopman算子能否捕捉非线性的极限环结构。

### 3.2 水箱系统 (`model_tank.m`)

单水箱液位控制系统的动力学方程通常由质量守恒定律给出：
$$ A \frac{dh}{dt} = q_{in} - q_{out} $$
其中 $A$ 是水箱横截面积，$h$ 是液位，$q_{in}$ 是流入流量（控制输入），$q_{out}$ 是流出流量。根据伯努利原理，流出流量通常与液位高度的平方根成正比：
$$ q_{out} = a \sqrt{2gh} $$
因此，系统方程为：
$$ \dot{h} = \frac{1}{A} q_{in} - \frac{a\sqrt{2g}}{A} \sqrt{h} $$
这是一个典型的非线性系统，在低液位时非线性尤为显著。

### 3.3 Otto 循环 (`model_otto.m`)

Otto循环由绝热压缩、等容加热、绝热膨胀和等容冷却四个过程组成。其状态变量通常包括气缸内的压力 $P$、体积 $V$ 和温度 $T$。

理想气体状态方程：
$$ PV = nRT $$
绝热过程（$PV^\gamma = \text{const}$）：
$$ T_2 = T_1 \left( \frac{V_1}{V_2} \right)^{\gamma - 1} $$
该模型涉及强烈的非线性热力学关系，是验证高级非线性控制算法的理想平台。

---

## 4. 控制与预测 (Control & Prediction)

### 4.1 线性预测 (Linear Prediction)

一旦通过EDMD或KDMD学习到了Koopman算子 $K$，系统的未来状态可以通过线性迭代预测：
$$ \Psi(x_{k+1}) = K \Psi(x_k) $$
$$ x_{k+1} = C \Psi(x_{k+1}) $$
其中 $C$ 是从特征空间映射回状态空间的矩阵（通常是 $\Psi$ 的逆或伪逆）。

### 4.2 Koopman 模型预测控制 (Koopman MPC)

由于提升后的系统 $\Psi(x_{k+1}) = K \Psi(x_k) + B u_k$ 是线性的（假设控制输入也以某种方式进入线性形式），我们可以构建如下的二次规划（QP）问题：

$$ \min_{u_0, \dots, u_{N-1}} \sum_{k=0}^{N-1} (\Psi(x_k)^T Q \Psi(x_k) + u_k^T R u_k) + \Psi(x_N)^T Q_f \Psi(x_N) $$
$$ \text{s.t.} \quad \Psi(x_{k+1}) = K \Psi(x_k) + B u_k $$
$$ u_{min} \le u_k \le u_{max} $$

这种方法结合了模型预测控制（MPC）处理约束的能力和线性系统计算的高效性，从而实现了实时的非线性控制。

---

## 5. 项目意义与应用场景 (Significance & Applications)

### 5.1 目的与意义
本项目的核心目的是开发一种**数据驱动的非线性控制框架**。
1.  **无需物理模型**：通过数据直接学习系统的动力学特征，避免了复杂的物理建模过程。
2.  **全局线性化**：与局部线性化（如Jacobian线性化）不同，Koopman算子提供了全局有效的线性描述。
3.  **实时性**：将非线性优化问题转化为凸优化（QP）问题，大大降低了计算复杂度，使得实时控制成为可能。

### 5.2 应用场景
1.  **复杂工业过程控制**：如化工反应釜、流体输送系统（Tank System），这些系统通常具有强非线性和不确定性。
2.  **能源系统**：如发动机控制（Otto Cycle）、风力发电机控制，优化效率并减少排放。
3.  **机器人与自动驾驶**：处理车辆动力学中的非线性特性，实现更精准的轨迹跟踪。
4.  **软体机器人**：由于难以建立精确的物理模型，数据驱动的Koopman方法非常适用。

---

**总结**：该项目通过Koopman算子理论将非线性动力学“提升”到线性空间，结合核方法处理高维特征，并利用降秩回归提取核心模态，最终实现高效、精确的非线性系统预测与控制。
