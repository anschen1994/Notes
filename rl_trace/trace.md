---
title: Trace, off-policy, on-policy
mathjax: true
date: 2020-06-30 16:21:56
tags: 强化学习
---
## 摘要
在这篇blog中，我们会介绍几个trace技巧，同时介绍研究人员如何用trace来把off-policy的数据纠正来计算关于on-policy的objective。

## TD($\lambda$) 和 Eligibility Traces [^4]
<!-- ### 简介 -->
众所周知，在强化学习中，我们需要估计value function，也就是要计算下面$G_t$的数学期望。
$$
G_t = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1}R_T
$$
那么很自然的有两种流派：
- 第一种就是每次完备地收集$R_{t+1},\cdots, R_T$这些数据，然后不断重复，最后计算出一个sample mean，这就是完全的基于return的方式。因为完备地获取数据需要不断的蒙特卡罗模拟整个过程，所以也称之蒙特卡罗方法。
- 第二种就是我们只模拟到某一步，剩下的部分用一个函数去近似，然后整个过程bootstrap。在最极端的情况下，就是只模拟一步，剩下全部用函数来近似:$G_t=R_{t+1}+\gamma V(S_{t+1})$

当然这边我们考虑一个比较general的做法，就是模拟 $n$ 步。我们可以定义$n$步return的概念，
$$
G_t^{t+n}=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n V(S_{t+n}) \tag{1}
$$
即所谓的中庸之道。当然因为$n$步的return在实际操作中的花费会比较大，因为每次都需要向前模拟$n$步。因此人们提出来了$TD(\lambda)$。

### $TD(\lambda)$
在方程.(1)的基础上，做进一步的改进，考虑一种特殊的线性结合的方式结合起来所有的$n$步return，例如，
$$
L_t(\lambda) = (1-\lambda)\sum_{n=1}^{+\infty}\lambda^{n-1}G_{t}^{t+n}
$$
对于在有限步$T$能够结束的游戏，我们做一个截断，
$$
L_t(\lambda) = (1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_{t}^{t+n} + \lambda^{T-t-1}G_t.
$$
通常意义上，人们称这种定义下的return为$\lambda$-return

### Eligibility Trace
从$\lambda$-return的定义，似乎比$n$步的return更加复杂，因为我们需要模拟出所有的游戏步骤。幸运的是，聪明的研究人员证明了下面的计算方式等价于上面的定义。为了更好的表达，我们引入Eligibility trace通过定义其更新方式:
$$
E_{t}(s) = \gamma\lambda E_{t-1}(s) + I_{s=S_t}
$$
在这些符号下，那么每步更新只需要TD error和eligibility trace，
$$
\Delta V_t(s) = \alpha \delta_t E_t(s), \delta_t = R_{t+1} + \gamma V_{t}(S_{t+1}) - V_t(S_t)。
$$
这边我们不做严格的数学证明。仅仅给出一个直觉上理解方式。让我们首先忘记强化学习这件事，考虑一个简单的case，我们想要估计一个硬币正面向上的数学期望通过不断抛硬币。假设在抛了$n$次之后，我们的估计结果是$p_n$，然后我们再抛一次，试图去更新估计。最trivial的方式就是，
$$p_{n+1} = (1-\lambda)p_n + \lambda I_{up}$$
这边$I_{up}$意味如果第$n+1$结果是向上，那么取1，否则取0。这不是和上面的过程有某种类似？现在我们思考eligibility trace做了什么，那么就比较简单了。其实就对于采样到的结果有了一个click，这边就是采取的加1的方式，然后随着时间的流失，这些结果的影响加上了discount。这边需要强调的discount并不具有真正意义时间序的关系，更多的是为了满足normalization，这样可以保证估计子的consistency。

## 使用各种trace来链接off-policy和on-policy

### Summary
在这个部分，我会考虑如何使用trace的技巧来弥补off-policy和on-policy的差距。比如我们的数据是从策略$\mu$中而来，但我们的目标是要估计$Q^{\pi}$而非$Q^{\mu}$,那我们需要怎么做呢。最简单的和我们之前所讨论一样使用importance sampling。假设我们有一个比较general的操作，
$$
\mathcal{R}Q(s,a):=Q(s,a)+E_{\mu}\left[\sum_{t\ge 0}\gamma^t (\prod_{s=1}^tc_s)(r_t+\gamma E_{\pi}Q(s_{t+1},\cdot)-Q(s_t,a_t))\right]
$$
- Importance Sampling [^3] ($c_s=\frac{\pi(a_s|s_s)}{\mu(a_s|s_s)}$)
显然我们可以纠正off-policy，但是这种做法有个问题，就是在一般情况下，$\frac{\pi(a_s|s_s)}{\mu(a_s|s_s)}$有很高的方差，就会导致算法很不稳定。
- Off-policy $Q^{\pi}(\lambda), Q^{*}(\lambda)$ [^5] ($c_s =\lambda$)
这种trace需要策略$\mu$不是那么off-policy。量化来讲，如果$\epsilon=\max_s\|\pi(\cdot|s)-\mu(\cdot|s)\|_1$，为了保证$Q^{\pi}(\lambda)$收缩性质，$\lambda < \frac{1-\gamma}{\gamma\epsilon}$; 为了保证$Q^{*}(\lambda)$的收缩性质，$\lambda < \frac{1-\gamma}{2\gamma}$。
- Tree-backup, TB($\lambda$) $(c_s=\lambda\pi(a_s|s_s))$
这种方式可以保证任意的off-policy都能收敛，但是很明显，如果$\mu$和$\pi$很接近，那么效率就会很低，因为$c_s$一直discount采样的效率，而在这种情况适合的$c_s$应该在$1$附近。
- Retrace($\lambda$)[^1] ($c_s=\lambda \min\left(1, \frac{\pi(a_s|s_s)}{\mu(a_s|s_s)}\right)$)
这个方法就是综合了上面三个方法的特点。
    - 给了trace一个bound，限制了方差
    - on-policy的时候，没有一直discount trace
    - off-policy的时候，限制了trace。
- V-trace [^2] ($c_s=\min(\lambda,\frac{\pi(a_s|s_s)}{\mu(a_s|s_s)})$)
V-trace除了用了一个不同的trace，还直接把temporal difference做了纠正，即映射用的是，
$$
\mathcal{R}Q(s,a):=Q(s,a)+E_{\mu}\left[\sum_{t\ge 0}\gamma^t (\prod_{s=1}^tc_s)\rho_t(r_t+\gamma E_{\pi}Q(s_{t+1},\cdot)-Q(s_t,a_t))\right],
$$
这边$\rho_t$是一个被截断IS纠正，$\rho_t=\min(\bar{\rho},\frac{\pi(a_t|s_t)}{\mu(a_t|s_t)})$并且$\bar{\rho}\ge\lambda \ge 1$。作者声称的一个好处是在完全on-policy的时候，V-trace可以完全复原也就是$\rho_t=c_s=1$，但是Retrace不可以$c_s=\lambda$。


### 数学定理
下面是两个重要的数学定理，从理论的角度保证了这些trace技巧的可行性。[^1]

#### Theorem 1
The operator $\mathcal{R}$ has a uniqued fixed point $Q^{\pi}$. If for each $a_s$ and history $\mathcal{F}_s$ we have $c_s=c_s(a_s,\mathcal{F}_s) \in [0, \frac{\pi(a_s|s_s)}{\mu(a_s|s_s)}]$, then for any function $Q$,
$$
\|\mathcal{R}Q-Q^{\pi}\| \le \gamma \|Q-Q^{\pi}\|
$$

#### Definition
We say that a sequence of policies $\pi$ is increasingly greedy a sequence $Q_k$ of Q functions if  the following property holds for all $k$: $P^{\pi_{k+1}}Q_{k+1} \ge P^{\pi_k}Q_{k+1}$, where operator $P$ is defined as,
$$
P^{\pi}Q(s,a) = \sum_{s',a'}\pi(a'|s')p(s'|s,a)Q(s',a')
$$

#### Thoerem 2
Consider an arbitrary sequence of behaviour policies $\mu_k$ and a sequence of target policies $\pi_k$ that are increasingly greedy w.r.t. the sequence $Q_k$:
$$
Q_{k+1} = \mathcal{R}_k Q_k
$$
where $\mathcal{R}_k$ is for $\pi_k, \mu_k$ and a **Markovian** $c_s=c(a_s,s_s)\in [0, \frac{\pi(a_s|s_s)}{\mu(a_s|s_s)}]$. Assume the target policies $\pi_k$ are $\epsilon_k$-away from the greedy policies w.r.t. $Q_k$, in the sense that $\mathcal{T^{\pi_k}}Q_k \ge \mathcal{T}Q_k -\epsilon_k\|Q_k\|e$, where $e$ is the vector with $1$-components, and
$$
\mathcal{T}^{\pi}Q:=r+\gamma P^{\pi}Q, \mathcal{T}Q := r + \gamma \max_{\pi}P^{\pi}Q
$$
Furthermore suppose $\mathcal{T}^{\pi_0}Q_0 \ge Q_0$. Then for any $k\ge0$,
$$
\|Q_{k+1}-Q^{*}\| \le \gamma \|Q_k -Q^*\| + \epsilon_k \|Q_k\|
$$
In consquence, if $\epsilon_k \to 0$, then $Q_k \to Q^*$.


## Reference
[^1] Munos, Rémi, et al. "Safe and efficient off-policy reinforcement learning." Advances in Neural Information Processing Systems. 2016.
[^2] Espeholt, Lasse, et al. "Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures." arXiv preprint arXiv:1802.01561 (2018).
[^3] Precup, Doina. "Eligibility traces for off-policy policy evaluation." Computer Science Department Faculty Publication Series (2000): 80.
[^4] Sutton, Richard S., and Andrew G. Barto. Introduction to reinforcement learning. Vol. 135. Cambridge: MIT press, 1998.
APA	
[^5] Harutyunyan, Anna, et al. "Q ($$\lambda $$) with Off-Policy Corrections." International Conference on Algorithmic Learning Theory. Springer, Cham, 2016.
APA	
