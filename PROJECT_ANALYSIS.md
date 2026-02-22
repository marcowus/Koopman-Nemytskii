# 项目原理与数学分析：基于Koopman-Nemytskii算子的非线性控制

**注意**：本项目复现了论文 **"Koopman-Nemytskii Operator: A Linear Representation of Nonlinear Controlled Systems" (arXiv:2503.18269)** 中的核心算法。该项目利用Koopman-Nemytskii算子理论，结合核方法（Kernel Methods）和降秩回归（Reduced Rank Regression, RRR），解决了非线性控制系统的辨识与控制问题。

针对您的问题：**Koopman-Nemytskii算子**并非仅指标准的Koopman算子，而是特指**带控制输入的Koopman算子**形式，强调其作为Nemytskii（或复合）算子作用于状态-控制对 $(x, u)$ 的函数空间上。相比传统的EDMD（扩展动态模态分解），本项目的独特之处在于引入了**降秩回归（RRR）**来估计算子，从而在高维核空间中实现了更鲁棒的正则化和特征提取，避免了标准最小二乘法的过拟合问题。

---

## 1. 核心概念：Koopman-Nemytskii 算子 (Koopman-Nemytskii Operator)

在传统的Koopman理论中，通常只考虑自主系统 $x_{k+1} = f(x_k)$。对于控制系统 $x_{k+1} = f(x_k, u_k)$，我们需要将算子的定义扩展到包含控制输入。

### 1.1 定义
考虑离散时间非线性控制系统：
$$ x_{k+1} = f(x_k, u_k) $$
其中 $x_k \in \mathcal{X}$ 是状态，$u_k \in \mathcal{U}$ 是控制输入。

**Koopman-Nemytskii 算子** $\mathcal{K}$ 定义在观测函数空间 $\mathcal{H}$ 上（例如 $L^2(\mathcal{X} \times \mathcal{U})$），作用于观测函数 $g: \mathcal{X} \times \mathcal{U} \to \mathbb{R}$：
$$ (\mathcal{K}g)(x_k, u_k) = g(f(x_k, u_k), u_{k+1}) $$
注意：这里通常需要对控制输入的演化做出假设（例如 $u_{k+1}$ 是 $u_k$ 的某种移位或通过控制器生成）。在简化的设置中，我们关注算子对状态观测值的预测能力，即寻找算子 $\mathcal{K}$ 使得：
$$ \Psi(x_{k+1}) \approx K \Psi(x_k, u_k) $$
这里 $\Psi(x, u)$ 是定义在状态-控制乘积空间上的特征函数（Observables）。

### 1.2 "Nemytskii" 的含义
Nemytskii算子（或复合算子）是指通过一个映射 $\phi$ 进行复合操作的算子 $C_\phi g = g \circ \phi$。在该项目中，强调 "Nemytskii" 是为了突显该算子通过将非线性映射 $f(x, u)$ 嵌入到函数空间中，从而将非线性控制问题转化为线性算子问题。

---

## 2. 核心算法：降秩回归 (Reduced Rank Regression, RRR)

这是本项目与标准Koopman/EDMD方法的主要区别，也是**"特别"**之处。

### 2.1 标准 EDMD 的局限性
标准的扩展动态模态分解（EDMD）试图求解最小二乘问题：
$$ \min_K \| \Psi_Y - K \Psi_X \|_F^2 $$
其中 $\Psi_X, \Psi_Y$ 是数据矩阵。这种方法在特征维度很高（尤其是使用核方法时）容易过拟合，且对噪声敏感。得到的算子 $K$ 往往是满秩的，包含了大量由噪声引起的虚假模态。

### 2.2 降秩回归 (RRR) 的引入
为了提取系统最主要的动力学特征并抑制噪声，本项目采用了**降秩回归（RRR）**。其目标是在约束算子秩的情况下最小化预测误差：
$$ \min_{K} \| \Psi_Y - K \Psi_X \|_F^2 \quad \text{s.t.} \quad \text{rank}(K) \le r $$

**数学推导与解法**：
令 $\hat{K}_{OLS} = \Psi_Y \Psi_X^\dagger$ 为普通最小二乘解。
RRR 的解 $\hat{K}_{RRR}$ 可以通过对加权预测协方差矩阵进行奇异值分解（SVD）得到。
1.  计算加权矩阵：$M = \Psi_Y \Pi_{\Psi_X} \Psi_Y^T$，其中 $\Pi_{\Psi_X}$ 是向 $\Psi_X$ 行空间投影的投影矩阵。
2.  对 $M$ 进行特征分解，取前 $r$ 个特征向量 $V_r$。
3.  RRR 解为：$\hat{K}_{RRR} = \hat{K}_{OLS} P_{V_r}$，即最小二乘解向主子空间的投影。

**RRR 的物理意义**：
-   **正则化**：通过限制秩 $r$，RRR 强制模型只学习数据中最显著的 $r$ 个动力学模式，滤除了高频噪声和无关特征。
-   **鲁棒性**：在核空间（Kernel Space）中，特征维度可能是无穷大的，直接应用EDMD会导致严重的数值问题。RRR 提供了一种在无穷维空间中寻找有限维低秩结构的有效方法。

---

## 3. 核方法 (Kernel Methods)

为了避免显式构造高维特征 $\Psi(x, u)$，项目使用了核技巧（Kernel Trick）。
定义核函数：
$$ k((x, u), (x', u')) = \langle \Psi(x, u), \Psi(x', u') \rangle $$
常见的选择是高斯核（RBF）或多项式核。
通过计算核矩阵 $G_{ij} = k(z_i, z_j)$（其中 $z_i = (x_i, u_i)$），我们可以在不显式计算 $\Psi$ 的情况下求解 RRR 问题。这对应了代码中的 `model1_kernel_input.m` 等文件。

---

## 4. 具体系统模型与实验

项目通过以下系统验证了 Koopman-Nemytskii + RRR 方法的有效性：

### 4.1 Landau-Stuart 方程 (`landau_stuart.m`)
-   **描述**：标准的非线性振荡器，具有稳定的极限环。
-   **作用**：验证 RRR 是否能准确捕捉到极限环的频率和幅值，以及在极限环附近的稳定性。

### 4.2 水箱系统 (Tank System, `model1`)
-   **方程**：$\dot{h} = \frac{1}{A} (q_{in} - a\sqrt{2gh})$
-   **挑战**：非线性流出项 $\sqrt{h}$ 在低液位时导致强非线性。
-   **结果**：文件 `model1_predict_RRR_grid41.png` 与 `DMD` 版本的对比，展示了 RRR 在预测精度和稳定性上的优势，特别是在数据稀疏或有噪声的情况下。

### 4.3 Otto 循环 (Otto Cycle, `model2`)
-   **描述**：热力学发动机循环，涉及复杂的状态方程 $PV=nRT$ 和绝热过程。
-   **挑战**：强非线性和混合动力学特性。

---

## 5. 总结：该项目的“特别”之处

针对您关于“和现有理论都一样”的疑问，该项目的创新点和特别之处在于：

1.  **明确的控制算子定义**：它不使用启发式的“扩展状态”方法，而是基于严格定义的 **Koopman-Nemytskii 算子** 理论，将状态和控制输入 $(x, u)$ 视为统一的函数空间上的变量。
2.  **降秩回归 (RRR) 正则化**：这是最核心的区别。大多数Koopman实现使用简单的最小二乘法（DMD/EDMD）。本项目引入 RRR，通过**低秩约束**来解决高维核空间中的过拟合问题。这在数学上等价于寻找一个最优的低维不变子空间，使得在该子空间内的线性预测误差最小。
3.  **核方法的结合**：将 RRR 与核方法（Kernel Methods）结合，使得算法能够处理无限维特征空间，同时保持计算上的可行性和数值稳定性。

综上所述，这个程序不仅仅是Koopman算子的简单应用，而是一个结合了**核方法**和**降秩统计学习**的高级非线性控制框架。
