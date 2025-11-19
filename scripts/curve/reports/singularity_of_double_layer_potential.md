# 二维双层势在边界上的跳跃条件推导

## 问题描述
考虑曲线积分：
$$D(x) = \int_{\Gamma} \frac{\partial}{\partial n_y} k(x-y) f(y)  dS_y$$
其中 $k(z) = \frac{1}{2\pi} \log |z|$ 是二维拉普拉斯方程的基本解。

## 跳跃条件推导

### 1. 局部坐标系建立
在 $x_0 \in \Gamma$ 处建立局部坐标系：
- $x_0$ 为原点
- $\Gamma$ 在 $x_0$ 处切于 $s$-轴
- 法向 $n$ 指向 $y$-轴正方向（外部）
- 参数化：$y(s) = (s, 0) + O(s^2)$
- 法向：$n_y \approx n = (0,1)$
- 密度函数：$f(y) \approx f(x_0)$

### 2. $x$ 不在 $\Gamma$ 上时的计算
令 $x = x_0 + \eta n = (0, \eta)$，则：
$$x - y \approx (-s, \eta), \quad |x-y|^2 \approx s^2 + \eta^2$$

法向导数：
$$\frac{\partial k(x-y)}{\partial n_y} = \frac{1}{2\pi} \frac{(x-y) \cdot n_y}{|x-y|^2} \approx \frac{1}{2\pi} \frac{\eta}{s^2 + \eta^2}$$

双层势近似：
$$D(x) \approx \int_{-\infty}^{\infty} \frac{1}{2\pi} \frac{\eta}{s^2 + \eta^2} f(s)  ds$$

#### 外部逼近 ($\eta \to 0^+$)
$$\lim_{\eta \to 0^+} \int_{-\infty}^{\infty} \frac{\eta}{s^2 + \eta^2} ds = \pi$$
$$\lim_{\eta \to 0^+} D(x) = \frac{1}{2\pi} \cdot \pi f(x_0) = \frac{1}{2} f(x_0)$$

#### 内部逼近 ($\eta \to 0^-$)
$$\lim_{\eta \to 0^-} \int_{-\infty}^{\infty} \frac{\eta}{s^2 + \eta^2} ds = -\pi$$
$$\lim_{\eta \to 0^-} D(x) = \frac{1}{2\pi} \cdot (-\pi) f(x_0) = -\frac{1}{2} f(x_0)$$

### 3. $x$ 在 $\Gamma$ 上时的计算
$$D(x_0) = \int_{\Gamma} \frac{\partial}{\partial n_y} k(x_0-y) f(y)  dS_y$$

局部展开：
$$y(s) = x_0 + s \tau + \frac{1}{2} k s^2 n + O(s^3)$$
$$x_0 - y = -s \tau - \frac{1}{2} k s^2 n + O(s^3)$$
$$(x_0-y) \cdot n_y \approx -\frac{1}{2} k s^2, \quad |x_0-y|^2 \approx s^2$$

法向导数：
$$\frac{\partial k(x_0-y)}{\partial n_y} \approx \frac{1}{2\pi} \frac{-\frac{1}{2} k s^2}{s^2} = -\frac{k}{4\pi}$$

积分收敛，等于Cauchy主值：
$$D(x_0) = \text{p.v.} \int_{\Gamma} \frac{\partial}{\partial n_y} k(x_0-y) f(y)  dS_y$$

### 4. 跳跃条件推导
比较极限值与直接值：

**外部问题：**
$$\lim_{\eta \to 0^+} D(x) = D(x_0) - \frac{1}{2} f(x_0)$$
$$D_{\text{ext}}(x_0) = \text{p.v.} \int_{\Gamma} \frac{\partial}{\partial n_y} k(x_0-y) f(y)  dS_y - \frac{1}{2} f(x_0)$$

**内部问题：**
$$\lim_{\eta \to 0^-} D(x) = D(x_0) + \frac{1}{2} f(x_0)$$
$$D_{\text{int}}(x_0) = \text{p.v.} \int_{\Gamma} \frac{\partial}{\partial n_y} k(x_0-y) f(y)  dS_y + \frac{1}{2} f(x_0)$$

## 数值计算补偿项
在边界元法中，当 $x$ 在 $\Gamma$ 上时：
- **外部问题**：补偿项为 $-\frac{1}{2} f(x)$
- **内部问题**：补偿项为 $+\frac{1}{2} f(x)$

即：
$$D_{\text{ext}}(x) = \text{p.v.} \int_{\Gamma} \frac{\partial}{\partial n_y} k(x-y) f(y)  dS_y - \frac{1}{2} f(x)$$
$$D_{\text{int}}(x) = \text{p.v.} \int_{\Gamma} \frac{\partial}{\partial n_y} k(x-y) f(y)  dS_y + \frac{1}{2} f(x)$$