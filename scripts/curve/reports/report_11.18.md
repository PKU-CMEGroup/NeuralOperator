

# Test results
上周的结果:

k=16, noact, L=10

| gradient | geo*w(x) |  geo*laplacian | Rel.train error| Rel.test error|
|----------|-------|----------------|----------------|---------------|
|          |       |                |      12.28%    |    13.22%     |
|   Yes    |       |                |      5.54%     |    5.78%      |
|          |  Yes  |                |      3.87%     |    4.25%      |
|          |       |       Yes      |      4.46%     |    5.24%      |
|   Yes    |  Yes  |                |      3.57%     |    3.99%      |
|   Yes    |       |       Yes      |      3.56%     |    4.01%      |
|          |  Yes  |       Yes      |      3.46%     |    4.07%      |
|   Yes    |  Yes  |       Yes      |      3.30%     |    3.84%      |

$$
geo = [n_x,n_y,fn_x,fn_y]
$$


更正后的结果:
$$
geo = [n_x,n_y,x,y]
$$

| gradient | geo*w(x) |  geo*laplacian | Rel.train error| Rel.test error|
|----------|-------|----------------|----------------|---------------|
|          |       |                |      12.28%    |    13.22%     |
|   Yes    |       |                |      5.54%     |    5.78%      |
|          |  Yes  |                |      4.94%     |    5.35%      |
|          |       |       Yes      |      4.73%     |    5.36%      |
|   Yes    |  Yes  |                |      4.55%     |    4.96%      |
|   Yes    |       |       Yes      |      4.00%     |    4.46%      |
|          |  Yes  |       Yes      |      3.80%     |    4.18%      |
|   Yes    |  Yes  |       Yes      |      3.67%     |    4.10%      |


$f*n$可能起作用的原因分析: split tests
```
x1 = speconv(fn, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
x2 = w(f)
geo_weight1 = self.softsign(geow1(geo))
x = x1 + x2 + geo_weight1*f
```
- geo = [n, x] , geo_weight1\*f ,  6.22%
- geo = [n, x] , geo_weight1\*fn,  ~8%
- geo = [n, f\*n] , geo_weight1\*f, 5.27%
- geo = [n, f\*n] , geo_weight1\*fn, 7.13%
- geo = [f\*n] , geo_weight1\*f, 6.06%
- geo = [f] , geo_weight1\*f, 6.35%


只加laplace, 不加geo
# Stokes kernel

- 方程：稳态不可压 Stokes 系统（$\mu>0$）
	$$
	\begin{cases}
	-\mu\,\Delta u + \nabla p = 0,\\
	\nabla\cdot u = 0.
	\end{cases}
	$$

- 核（自由空间基本解，$r=x-y$）：
	- 3D：
		$$
		G_{ij}(x,y)=\frac{1}{8\pi\mu}\Big(\frac{\delta_{ij}}{|r|}+\frac{r_i r_j}{|r|^3}\Big),\qquad
		\Pi_{j}^{3D}(x,y)=\frac{1}{4\pi}\,\frac{r_j}{|r|^3}.
		$$
	- 2D：
		$$
		G_{ij}(x,y)=-\frac{1}{4\pi\mu}\Big(\delta_{ij}\,\log |r| - \frac{r_i r_j}{|r|^2}\Big),\qquad
		\Pi_{j}^{2D}(x,y)=\frac{1}{2\pi}\,\frac{r_j}{|r|^2}.
		$$

- 解的表达式（单层势，$x\notin\Gamma$）：令 $\Gamma$ 为闭曲线/曲面，$q:\Gamma\to\mathbb R^d$，则
	$$
	u_i(x)=\int_\Gamma G_{ij}(x,y)\,q_j(y)\,\mathrm dS_y,\qquad
	p(x)=\int_\Gamma \Pi_j(x,y)\,q_j(y)\,\mathrm dS_y.
	$$
	其中 $q$ 的取值由边界条件确定：
	- 若给定速度（Dirichlet），用方程 $\,\mathcal S q = u|_\Gamma\,$ 求 $q$；



## 数据生成

我们生成
$$
k(x,y) = \frac{r_1r_2}{r^2}
$$
对应的曲线数据来测试, 固定L=14, layers = [128,128], noact, 测试结果如下

| kmax | $f*n$ | gradient | geo*w(x) | Rel.train error | Rel.test error |
|------|-------|----------|----------|------------------|-----------------|
| 16   | Yes   |          |          | 3.83%            | 4.23%           |
| 16   | Yes   | Yes      |          | 2.79%            | 3.13%           |
| 16   | Yes   |          | Yes      | 1.24%            | 1.38%           |
| 16   | Yes   | Yes      | Yes      | 1.37%            | 1.49%           |
| 16   |       |          |          | 4.17%            | 4.57%           |
| 16   |       | Yes      |          | 3.96%            | 4.31%           |
| 16   |       |          | Yes      | 1.53%            | 1.64%           |
| 16   |       | Yes      | Yes      | 1.52%            | 1.63%           |
| 32   |       |          |          | 1.43%            | 1.71%           |
| 32   |       |          | Yes      | 0.53%            | 0.66%           |
| 32   |       | Yes      |          | 1.37%            | 1.63%           |
| 32   |       | Yes      | Yes      | 0.55%            | 0.68%           |
