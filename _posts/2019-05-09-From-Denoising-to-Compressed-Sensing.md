---
layout: post
comments: true
title:  "From Denoising to Compressed Sensing"
excerpt: "-"
date:   2019-05-09 12:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
---

[TOC]

**由于本文内容实在有点难，因此首先从BM3D方法开始，前半部分暂时无限期停更**

## 从降噪到压缩感知

### 摘要

降噪算法的目的是去除信号中的噪声、误差以及扰动等。在过去的几十年里，人们对相关算法做了广泛研究，结果是先进的降噪算法那可以有效地去除大规模加性高斯白噪声。压缩感知重建算法的目的是基于小数目的随机测量重建一个结构化信号。典型的压缩感知重建算法可以从有扰动的观测中迭代地估计一个信号。本文回答了一个自然的问题，即如何有效地在压缩感知重建中部署一个通用的降噪算法？为了回答这个问题，我们开发了一个近似消息传递（`Approximate Message Passing， AMP`）框架的扩展，叫做`Denoising-based AMP, D-AMP`，可以在迭代过程中继承大量的去噪器。

我们测试了，当与高性能去噪器一起使用的时候，我们的压缩感知算法能够提供目前最好的重建效果，而且速度比那些先进算法快几十倍。我们分析了一些理论原因来解释这些卓越性能的现象，这个的关键原因应当是算法中的去噪器在迭代中采用了合适的`Onsager`校正项，使得每一次迭代中，信号中的扰动非常近似高斯白噪声的分布，从而易于去除。

## I - 引入

### A. 压缩感知

压缩感知重建算法面对的基本挑战是，在小数量的测量下重建高维信号。压缩测量的过程可以被认为是线性映射，从长度为$$n$$的信号向量$$x_0$$映射到一个长度为$$m\ll n$$的测量向量$$y$$。由于这个过程是线性的，因此可以被建模为测量矩阵$$\boldsymbol \Phi \in \mathbb C^{m\times n}$$。这个矩阵可以采用各种物理解释，例如在压缩采样`MRI`中，这个矩阵可以是$$n\times n$$的傅里叶矩阵的采样行。在单像素相机中，这个矩阵可以是代表微镜阵列调制的序列。

往往信号$$x_0$$在某些变换域中是稀疏的，或者近似稀疏的，例如$$x_0=\boldsymbol \Psi u$$，其中$$\Psi$$是逆变换矩阵，$$u$$是变换域中稀疏信号。在这种条件下，我们将测量与转换集中到一个矩阵中$$\boldsymbol A=\boldsymbol \Phi\boldsymbol \Psi$$。当我们不采用稀疏性假设的时候，我们认为$$\boldsymbol A=\boldsymbol \Phi$$。在后文中我们的测量阵都采用$$\boldsymbol A$$来表示。

压缩感知重建问题是求解欠定方程组问题。原信号$$x_0$$经过测量后得到$$y=\boldsymbol Ax_0+\omega$$，压缩感知的问题就是根据得到的$$y$$计算$$x_0$$。由于$$m\ll n$$，所以这个问题是一个十分欠定的问题。因此还原信号的第一假设就是$$x_0$$是有特殊结构的信号，然后在满足$$y\approx \boldsymbol Ax$$的$$x$$搜索符合给定结构的解。对于$$x_0$$稀疏的条件下，一种常见的解法是凸优化问题
$$
\begin{equation}
\begin{split}
\underset{x}{argmin}\left \| x \right \|_1\\
&subject\: to \left \| y-Ax \right \|_2^2\leq\lambda
\end{split}
\tag{1}
\end{equation}
$$

这被称为基本追踪去噪`basis pursuit denoising, BPDN`，当$$x_0$$充分稀疏，且$$\boldsymbol A$$满足一些条件的时候，这种方法可以有效重建$$x_0$$

一开始人们也是用凸优化的方法解决压缩感知重建问题，然而档处理大信号的时候，例如图像信号，图优化算法计算复杂度就过高了。有关相关算法细节内容可以自己在参考文献中查询，本文可以从`IEEE Transactions on IF`找到。

### B. 主要贡献

稀疏模型对于许多信号来说已经十分精确了，而且已经是压缩感知领域的研究重点。遗憾的是，以稀疏假设为基础的方法对于图像来说应用并不好。原因大概是自然图像在任何已知的方法中并不具备精确的稀疏表达（即使是采用离散余弦变换、小波以及其他）。图像展示的是经典图像的小波变换的系数，显然是一个非稀疏的系数分布直方图。许多系数都是非零的，甚至有很多远离$$0$$的系数。因此寻求小波稀疏性的算法并不能恢复这个信号。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/dfig1.jpg" style="display: inline-block;" width="500"/>
</div>

为了应对这样的失败，研究者提出了许多其他的压缩感知的重建描述模型。这些详细内容这里将不做介绍。本文中我们采用一种补充方法来增强非稀疏信号的压缩感知重建水平。相比关注于设计新的信号模型，我们测试了现有的去噪算法对压缩感知重建效果的增强。我们的想法很简单，信号去噪算法（无论是基于显式还是隐式模型）已经被设计并完善了数十年了，因此采用去噪算法的重建方案应当可以捕获迄今为止尚不能被现有压缩感知算法所捕获的结构。

`其实我觉得这个脑回路不行`

这里介绍`Approximate Message Passing, AMP`算法，近似消息传递算法，提供了一个自然的方法，在压缩感知重建中采用了去噪算法。我们称采用了`D`去噪器的`AMP`算法为`D-AMP`算法。`D-AMP`算法假设$$x_0$$属于一类信号$$C\subset \mathbb R^n$$，例如存在去噪器$$D_\sigma$$指定大小的自然图像类。每一个去噪器$$D_\sigma$$可以被应用于$$x_0+\sigma_z$$，其中$$z\sim N(0,I)$$，且可以返回一个估计值$$\hat x_0$$，（我们至少希望这张图）比之前的$$x_0+\sigma_z$$要更接近原图像。去噪器可以采用简单结构或其他复杂结构，将在后文讨论。本文中我们只是把每个去噪器看成是黑箱，他们收到叠加高斯白噪声的信号，然后返回对信号的估计。因此我们不考虑这个算法内部究竟是如何实现的。这是我们的推导适用于各种信号和各种去噪器。

上面所说，结合了去噪器的`AMP`算法有若干现有压缩感知重建算法所不具备的好处

- 很容易将一个算法部署到多种类型的信号问题上
- 比现有算法性能好，而且对测量噪声有很强的鲁棒性
- 带有一个描述框架，不仅可以知道算法的上限，还可以知道将来该如何选择

`D-AMP`算法如下迭代部署
$$
\begin{equation}
\begin{split}
x^{t+1}&=D_{\hat \sigma^t}(x^t+\boldsymbol A^*z^t) \\
z^t&=y-\boldsymbol Ax^t+z^{t-1}div D_{\hat \sigma^{t-1}}(x^{t-1}+A^*z^{t-1})/m\\
(\hat\sigma^t)^2&=\frac{\left\| z^t \right\|_2^2}{m}
\end{split}
\tag{2}
\end{equation}
$$
其中$$x^t$$是每次迭代中对$$x_0$$的估计，$$z^t$$则是对残差的估计。我们后面将推导出$$x^t+\boldsymbol A^*z^t=x_0+v^t$$，此处$$v^t$$可以看做是独立同分布的高斯噪声。$$\hat\sigma^t$$是噪声标准差估计，$$div D_{\hat \sigma^{t-1}}$$表示去噪器的散度（此处仅仅是偏导数的和），其中的$$z^{t-1}div D_{\hat \sigma^{t-1}}(x^{t-1}+A^*z^{t-1})/m$$就是所谓的`Onsager`修正项。我们将在后面展示这一项产生的性能影响。这个修正项的计算不是完全明确的，许多著名的去噪器并没有明确地计算公式，然而我们将证明这一项可以近似计算，而不用管究竟是什么形式的去噪器。

`D-AMP`将现有的去噪算法应用于压缩测量的向量。直觉上，每一次迭代中`D-AMP`将获得更好的估计结果，并且该估计序列最终是收敛于$$x_0$$的（无偏估计）。

为了预测`D-AMP`算法的性能，我们使用了一个新式的状态演化框架，在理论上跟踪算法每次迭代时的噪声和标准偏差$$\hat \sigma_t$$。我们的框架扩展并验证了前人文献中的框架，通过大量的仿真，我们证明了在高维场景中，我们的状态演化框架准确预测了算法的均方误差`MSE`。基于我们的状态演化，我们描述`D-AMP`算法性能，并将算法重建新号所需要的采样点数与降噪器的性能联系起来。我们还采用了状态演化去跟踪了微调去噪器参数的影响 以及 算法对测量噪声的敏感度。另外还用这个框架探索了算法的最优性。下图展示了一个示范结果，采用非去噪算法（利用小波域稀疏性）和去噪算法产生的不同。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/dfig3.jpg" style="display: inline-block;" width="500"/>
</div>

### C. 相关工作

#### Approximate Message Passing and Extensions

近似消息传递及其扩展算法。

在过去的五年里，消息传递和近似消息传递算法在压缩感知中广泛研究的主题。许多已经发表的文章都采用了贝叶斯框架，这种框架中信号的先验概率$$p_x$$已经被定义在了信号$$x_0$$的分类`C`中了。消息传递算法被视为是一种计算后验均值$$\mathbb E(x_0\mid y,A)$$的方法。消息传递算法已经被简化为了近似消息传递算法，此处是利用了数据的高维度信息。状态演化框架已经被作为一种分析近似消息传递算法的方法提出。

我们的工作与上述不同之处在于，我们不对信号的先验做出任何假设，这将导致我们的状态演化框架出现不同，后面会详述差异。

值得一提的是，在开发`D-AMP`算法时我们不关心是否算法是否在指定的某个先验的条件下是否会接近某一个后验，也不在乎`D-AMP`算法中采用的去噪器是否与某个先验相关。相反，我们只关心一个重要因素，即$$x^t+\boldsymbol A^* z^t-x_0$$是否接近独立同分布高斯噪声。我们的分析是基于这个假设的，后面也会有仿真。

之前也有人做过类似的分析，在他们的假设中，去噪器`D`可以是任何满足尺度不变性的函数，我们的不同点在于

- 我们不对尺度不变性做要求，因为实际中很多去噪器都不满足这个
- 我们对方法与状态演化过程做了更加细致的验证
- 提出了算法参数调整原则
- 在算法中如何使用没有明确功能形式的去噪器
- 我们调查了算法的最优性

本文中提出的状态演化模型可以确实的跟踪算法的性能。

在本文完成时，另一篇关于如何在算法中部署去噪器的文章发表了，本文将进一步探索这个方法在压缩重建中的应用。

#### Model Based CS Imaging

许多研究者注意到了基于稀疏性算法的缺陷，也探索了其他的更为复杂的信号模型。这些模型可以通过约束解空间的方法显式执行，也可以采用惩罚函数的方法隐式地执行。

`2019-5-10 14:09:36`

算了这部分先不看了，直接去后面算了

## II - 基于去噪声的近似消息传递

我们考虑一族对于信号类别$$C$$的去噪算法$$D_\sigma$$。我们的目标是采用这些去噪器去从$$y=\boldsymbol Ax_0+w$$获取一个$$x_0\in C$$的良好估计（其中$$w\sim N(0,\sigma_x^2 I)$$）。我们从以下方法开始，这个方法是受到迭代硬阈值算法及其扩展形式在基于块的压缩图像的应用的启发。为了更好地理解这个算法，我们考虑无噪声条件下的$$y=\boldsymbol Ax_0$$，此时假设去噪器是一个投影（线性变换）。由$$y=\boldsymbol Ax$$和集合$$C$$定义的仿射子空间如图所示，我们假设$$x_0$$是唯一解，即二者交集的唯一元素。



<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/dfig4.jpg" style="display: inline-block;" width="500"/>
</div>
我们知道，解在投影子空间$$\{ x\mid y=\boldsymbol Ax \}$$，因此我们从$$x_0=0$$开始，向与子空间正交的方向移动，例如$$\boldsymbol A^*y$$。这个方向与子空间较为接近，但是却不一定是接近$$C$$的，于是，我们利用去噪（或者说投影）去获取一个估计来满足信号分类$$C$$的结构。整个原理就像图里所说的。

经过以上步骤我们得到了$$D(A^*y)$$。就像我们用图像表示的一样，重复以上两个步骤，在梯度方向移动并向$$C$$投影，我们的估计最终收敛到$$x_0$$。算法表示为
$$
\begin{equation}
\begin{split}
x^{t+1}&=D_{\hat \sigma}(x^t+\boldsymbol A^*z^t) \\
z^t&=y-\boldsymbol Ax^t\\
\end{split}
\tag{3}
\end{equation}
$$

为了便于表示，我们引入了估计残差的向量，即$$z^t$$，我们称这种算法为基于去噪的迭代阈值处理（`denoising-based iterative thresholding, D-IT`）。如果我们改变去噪器（在上图中仅相当于一个到$$C$$的投影），让他是一个真正的去噪函数，我们隐含地假设$$x^t+\boldsymbol A^*z^t$$可以表示为$$x_0+v^t$$，其中$$v^t\sim N(0,(\sigma^t)^2I)$$，而且与$$x_0$$相互独立。于是通过采用一个去噪器，我们获得了一个更为接近$$x_0$$的信号。不幸的是，这种假设对于`D-IT`而言并不成立，这与我们在前文中观察到的用于迭代软阈值处理的现象相同。

在迭代阈值算法的条件下，我们提出的避免噪声的非高斯性解决方案是采用消息传递/近似消息传递。遵循相同的路径，我们提出以下算法。

$$
\begin{equation}
\begin{split}
x_{\cdot \to a}^{t+1}&=D_{\hat \sigma^t}\left(\left[ \begin{matrix} \sum_{b\neq a}\boldsymbol A_{b1}z_{b\to 1}^t \\ \sum_{b\neq a}\boldsymbol A_{b2}z_{b\to 2}^t \\ ...\\ \sum_{b\neq a}\boldsymbol A_{bn}z_{b\to n}^t \end{matrix} \right]\right)\\
z_{a\to i}^t&=y_a-\sum_{j\neq i}\boldsymbol A_{aj}x_{j\to n}^t
\end{split}
\tag{4}
\end{equation}
$$
在这里，$$\hat x$$提供了一个$$x_0$$的估计。$$\hat \sigma^t$$表示下式标准差。
$$
\begin{equation}
\begin{split}
v_{\cdot \to a}^t=\left(\left[ \begin{matrix} \sum_{b\neq a}\boldsymbol A_{b1}z_{b\to 1}^t \\ \sum_{b\neq a}\boldsymbol A_{b2}z_{b\to 2}^t \\ ...\\ \sum_{b\neq a}\boldsymbol A_{bn}z_{b\to n}^t \end{matrix} \right]\right)-x_0
\end{split}
\tag{5}
\end{equation}
$$
我们在上面总结过，上式在高维中与高斯噪声很接近（此时`m`和`n`都很大 ）。对于一类标量去噪器，这一结果已经得到了严格证明，并且也可以用于块状去噪器。

尽管它们在避免有效噪声向量$$v$$的非高斯性方面有一定的优势，但是消息传递算法有$$m$$次测量，也因此给出了这么多不同的$$x_0$$估计；相似的，每一个噪声向量都有$$n$$个估计结果。这些需要很高的计算量，但是幸运的是，如果问题是高维的，我们可以近似消息传递算法的迭代过程，并获得基于去噪的近似消息传递算法
$$
\begin{equation}
\begin{split}
x^{t+1}&=D_{\hat \sigma^t}(x^t+\boldsymbol A^*z^t)\\
z^t&=y-\boldsymbol Ax^t+z^{t-1}\frac{div D_{\hat \sigma^{t-1}}(x^{t-1}+\boldsymbol A^*z^{t-1})}{m}
\end{split}
\tag{6}
\end{equation}
$$
上述的`D-IT`与`D-AMP`算法的唯一不同在于修正项$$z^{t-1}\frac{div D_{\hat \sigma^{t-1}}(x^{t-1}+\boldsymbol A^*z^{t-1})}{m}$$。

有关`D-AMP`从`D-MP`的推导与`MP`中推导`AMP`相似。与`D-IT`相似，`D-AMP`依赖于有效噪声$$v^t=x^t+\boldsymbol A^*z^t-x_0$$近似于独立同分布高斯噪声的假设。我们的经验结果证明了这一个假设，如图所示`D-AMP`迭代，使用的降噪算法是`BM3D`。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/dfig5.jpg" style="display: inline-block;" width="500"/>
</div>
这里先介绍一下啥是`BM3D`算法。



---

---

---

## BM3D 算法原理与实现

### 原理

- `2019-5-14 22:07:53`不管怎么看都觉得这个算法不过是一个稍加改进的变换域阈值操作。就这还能出结果真是了不得。

---

`BM3D`算法是目前一种较好的去噪重建算法，在对噪声没有先验的条件下进行噪声去除。

主要流程如下

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/BM3D.jpg" style="display: inline-block;" width="650"/>
</div>

*稍微吐槽一下，现在能找到的`BM3D`算法都是封装后的，根本不是开源代码。*

原理部分。我们首先说算法实现，然后解释原因。

这个算法分为两步，分别是基础估计$$Basic\:Estimate$$和$$Final \:Estimate$$。

- `Basic Estimate`
    - 逐块估计。对于含噪声图像的每一个`Block`，我们
        - 分组。寻找与当前处理的块相似的块，并把他们堆在一起，组成一个`3D`阵列。
        - 协同硬阈值`Collaborative hard-thresholding`。对上述分组结果进行某种`3-D`变换，对变换系数进行**硬阈值**处理来降低噪声。硬阈值处理是一种阈值处理，可以对小于阈值的输入表现为无输出（全阻），而对于大于某阈值的结果表现为全输出（全通）。通过处理后的结果经过逆变换得到原图像。注意硬阈值的处理会丢失很多细节。我们以图像的傅里叶变换为例，图像的傅里叶变换并不是严格意义上的稀疏的，如果对较少的高频部分进行硬阈值处理，我们得到的图像就会丢失掉细节信息。
    - 聚合。通过上一步得到的反变换结果，我们重建原图像。一般是根据所有得到的逐块估计进行加权平均得到原图，至于如何加权平均我们后面会说明。
- `Final Estimate`，利用上一部的估计结果，我们使用维纳滤波的方式获得更好的结果。
    - 逐块估计。
        - 分组。在第一步估计中，我们用块匹配的方法进行了分组。使用这些位置，我们再构造一个`3-D`阵列，一个是来自原来的有噪图像的，一个是基本估计后的，
        - 协同维纳滤波`Collaborative Wiener filtering`。对上述两个组进行`3-D`变换，对于含噪图像分组我们采用维纳滤波的方式，我们此处认为基本估计得到的图像功率谱就是真正的功率谱。对滤波后的结果我们进行逆变换得到的结果就被认为是真实估计结果。
    - 聚合。计算真实图像的最终估计，采用加权平均的方式。

以上就是算法实现的整个流程了，我们今后尝试实现一个一维的`demo`。我们现在先来解释一下每一步的出发点与目的。

#### A. 分组

这个算法首先采用了分组的手段进行处理。分组这里可以通过很多手法实现，例如K均值聚类，自组织映射等，不一一介绍。

通常我们将信号片段之间的相似度计算为某个定义下距离的倒数。因此，较小的距离就可以代表着高相似度。我们可以采用各种距离测量手段，例如两个信号片段之间的范数误差作为距离，当然也可以使用加权欧几里得距离等其他方法。当处理一些复杂或者不确定（随机）信号的时候，我们可能必须要先提取信号中的某些特征，然后进行**特征**的**距离**的比对。

#### B. 分组匹配

这篇论文里面水也很多，前面那一段就是白开水白瞎我翻译，实际上就是说，虽然有很多分组算法，但是往往是分出几个`cluster`，互相之间并不严格相关，没啥参考意义。他这里就是胡扯，都无监督自分类了，还扯这么多有的没的。通过直接聚类方法得到的分组可能在重建图像上需要高计算复杂度的迭代过程。关于图像的分割与重建过程中出现的问题，其实已经有了很多的研究，这里不再赘述各种原理，我们直接说结论，想要有利于图像的重建，我们可以采用匹配的方法进行分块。我们在不同位置进行分块，然后对比块的相似性，将相同的块分到一组。一般来说分到一组的是距离差异小于指定阈值的块，当然一般会设置最大匹配数量。

块匹配方法是一种特殊的匹配方法，已经广泛应用于视频压缩中的运动估计等。该方法就是找到相似的块，并将其堆叠为`3-D`阵列。图中显示了块匹配方法，

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/BM.jpg" style="display: inline-block;" width="650"/>
</div>

标记了`R`的部分是参考快，其他是与之匹配的块。需要提到的是，任何块都可以成为参考块`R`。

#### C. 联合滤波（协同滤波）

给定一个有`n`个片段的分组，这个分组经过联合滤波会产生`n`个不同的估计，每个片段都能获得一个估计。一般来说这些估计都是不同的。下图是一个示意图，是分组的实例。这是一个理想的情况，假设图像被叠加的高斯白噪声并没有画出。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/BM3Dfig2.jpg" style="display: inline-block;" width="650"/>
</div>

我们关注已经被块匹配处理后的结果，我们可以看出在理想状态下这些分组完美地匹配了，这样我们就可以认为，按元素平均成为了一种良好的估计器`estimator`。关于估计器的说法请移步谱估计。因此对于每个组，这种平均了所有的估计结果。如果我们假设无噪声块是相同的，因此估计一定是无偏的，因此最终估计误差仅归因于残差方差，其余组中块数成反比。良好估计的前提是真的有足够多的可以相互匹配的块。

然而在真正的自然图像中完美匹配的块并不多。如果不同的片段在同一分组中，那么逐块估计的结果就是有偏的。因此我们需要一种更好地协同滤波方法。

#### D. 变换域收缩协同滤波

在变换域中的收缩算法可以是一种更好地协同估计方法。我们假设$$d+1$$维的分组已经被构造了，协同收缩方法就是如下步骤

- 对$$d+1$$维的块进行一种线性变换。
- 收缩变换系数来降低噪声（例如软阈值与硬阈值等方法）。
- 进行线性逆变换来产生每一组的片段估计。

上述协同变换域收缩方法在处理自然图像的时候是尤其有效的，例如我们`B`中的那张自然图像。他们的每一个分组的特点是，

- 分组的每个片段中，片段内的像素之间存在相关性，这是自然图像的特点。
- 不同片段之间的相同位置像素存在相关性，这时我们块相似匹配算法带来的特点。

我们这里采用的$$3-D$$变换方法可以利用上述两种相关性，因此对真实信号产生一个稀疏的表示。这个稀疏特点让收缩变得十分高效，既能去除噪声，又能保护图像的特征。

---

简单介绍一下这种协同收缩算法的优点。假设不用这个方法，对于已经分组以后的结果我们进行`2-D`变换，单独对每一个含有`n`个片段的块进行变换。由于这些分组块内都是十分相似的，因此对于每一次的变换我们得到的重要系数的数量，例如我们说是$$\alpha$$个，都是相似或者相同的。这表示我们的`n`个片段，最终将得到$$n\alpha$$个重要系数，用这些系数来代表这个`block`。

相反，我们采用协同滤波方法中，我们除了应用`2-D`变换以外，还在分组块上应用了一个`1-D`变换，等价于对整个组应用了可分离的`3-D`变换。如果这个`1-D`变换有以`DC`为基础的元素（这个DC啥意思？），那么由于块内信号的高度相似性，那么我们几乎只用$$\alpha$$个参数就能较好地代表整个组的信号。那么这样就强化了信号的稀疏性，稀疏程度随着同组块的增加而增加。

就像第一个图中的结果一样，小的块之间是高度相似的，这种不同位置的高度相似小片段在自然图像中是很常见的，在自然图像处理的过程中，我们对图像存在相似块的假设是显而易见可以使用的。

在我们的算法实现中，尤其是第二步，我们主要是受到了两个启发创新

- 使用基础估计首先估计一张去噪结果，用以改善分组结果
- 使用基本估计做过维纳滤波的导频信号可以更有效准确地实现硬阈值处理`3-D`变换谱。

维纳滤波器是要根据一定的先验结果进行的，其目的是最小化均方误差。由于在实际应用中我们的先验知识少的多，因此实现维纳滤波只能通过各种估计的方法。

数学上，我们可以用如下方式解释。首先，我们认为我们观察到的图像是
$$
\begin{equation}
\begin{split}
z(x)=y(x)+\eta(x)
\end{split}
\tag{1}
\end{equation}
$$
其中$$x$$是图像域的`2-D`空间坐标（整数），$$y$$是真实图像，后面附加的是独立同分布的高斯白噪声，方差已知。对于图像而言，我们选取$$N_1\times N_1$$大小的块，用$$z$$代表块，用$$x$$代表左上角的坐标。所以我们说，$$Z_x$$是$$z$$图的$$x$$位置块。一组`2-D`图像块可以用粗体大写字母表示，例如$$\boldsymbol Z_S$$代表一个`3-D`阵列，由$$Z_x$$块组成，位于$$x$$位置。为了区别两步操作中不同的符号，我们使用$$ht$$和$$wie$$角标来表示。

#### A. 逐块估计

块匹配过程采用滑动窗口方法。整个处理过程分为

- 联合硬阈值处理
- 联合维纳滤波处理

结果的估计我们称为**逐块估计**。实际上最上面的框图里面，这两个操作是完全相同的。对于参考图像，我们称其为$$Z_{XR}$$。和前面介绍的一样，分组的时候，只有像素之间的距离小于某一个指定阈值的时候我们才会认为两个块可以相互匹配。这里就是传统的二范数差。
$$
\begin{equation}
\begin{split}
d^{noisy}(Z_{x_R},Z_x)=\frac{\left \| Z_{x_R}-Z_x \right \|_2^2}{(N_1^{ht})^2}
\end{split}
\tag{2}
\end{equation}
$$
如果块并不重叠，那么这个距离就是非中心化卡方分布随机变量，其均值为
$$
\begin{equation}
\begin{split}
\mathbb E\left[ d^{noisy} \right]=d^{ideal}+2\sigma^2
\end{split}
\tag{3}
\end{equation}
$$
方差比较复杂，这里不进行计算。方差的增长趋势是$$\mathbb O(\sigma^4)$$的，因此对于较大的方差和较小的块，该算法的重建效果将变差。所以我们才会选择采用块匹配的方法尽量去匹配接近的图像，使方差减小。

为了避免方差增大的这个问题，我们建议使用简单的预滤波来处理一下，在不同的块上应用归一化的`2-D`线性变换，对获得的系数进行硬阈值处理来实现。
$$
\begin{equation}
\begin{split}
d'(Z_{x_R},Z_x)=\frac{\left \| \mathbb Y'(\mathbb T^{ht}_{2D}(Z_{x_R}))-\mathbb Y'(\mathbb T^{ht}_{2D}(Z_x)) \right \|_2^2}{(N_1^{ht})^2}
\end{split}
\tag{4}
\end{equation}
$$
其中$$\mathbb T$$是归一化二维线性变换，$$\mathbb Y$$是硬阈值处理。该块的结果是一个集合，集合内是与参考块图像相似的块的坐标。坐标统一为左上角像素。（我们不做反变换，只是在变换域内进行距离比对，当这个变换时正交变换的时候，距离在空间域的描述和变换域内的描述是一致的）

**提醒一下这个集合是存放位置的。**
$$
\begin{equation}
\begin{split}
S=\left\{ x\in X:d(Z_{x_R},Z_x)\leq\tau_{match}^{ht} \right\}
\end{split}
\tag{5}
\end{equation}
$$
后面这是一个常数阈值，描述两个块之间的距离。匹配过程中自己与自己的匹配也是考虑在内的，因此分组结果都至少为$$1$$个每组。匹配后的块组成了`3-D`阵列，大小为$$N_1^{ht}\times N_1^{ht}\times \mid S \mid$$。这里我懒得用其他符号来规定组内成员数了，记住这个$$S$$在不同组是不同的。每一个`3-D`阵列我们命名为$$\boldsymbol Z_{S^{ht}_{x_R}}$$。这个人起名字真的傻逼。做块匹配的时候，由于全图遍历的原因，最后肯定会出现大规模的重叠，这个我们并不做限制。

这个`3-D`阵列的硬阈值处理可以在`3-D`变换域中完成，选取的变换方式应是归一化的三维线性变换，定义为$$\mathbb T_{3D}^{ht}$$，应该能够利用前面讲到的两类相关性（块内相关性和组内块间同位置相关性），并且能保证变换结果有着较为优秀的稀疏性。这将能够使硬阈值处理有着更为有效的噪声抑制，反变换得到估计阵列。估计结果表达式就不写了，变换、硬阈值、反变换即可。注意反变换结果也仍然是$$\mid S \mid$$个`block`。

以上实际上都是第一步的详细说明，下面进行第二步的详细说明。

第一步主要就是变换域的硬阈值处理，第二步是在变换域中做维纳滤波。我们通过上一步已经得到了基础估计`basic estimate`结果，我们的降噪结果可以通过利用这个结果的维纳滤波来改善。我们首先**假设在第一步的基本估计中噪声被显著削弱了**，我们在分组的时候重新利用**基础估计**来进行分组。距离公式由含噪图像的计算变成了去噪图像的计算。这样的计算就是对理想图像分组的一个更为精确的估计。**提醒一下这个集合是位置的集合**。我们通过基础估计分组，相同的分组方式用于原来的含噪图像，然后进行维纳滤波。到此为止这个算法的基本思想我们就`get`了。
$$
\begin{equation}
\begin{split}
S=\left\{ x\in X: \frac{\|\widehat Y^{basic}_{x_R}-\widehat Y_x^{basic}\|_2^2 }{(N_1^{wie})^2} \leq\tau_{match}^{wie} \right\}
\end{split}
\tag{6}
\end{equation}
$$
我们此处的分组要分两组，一个是通过基础估计分组，一个通过噪声图像分组。

- 通过基础估计的分组，将基础估计中相似的块进行块匹配。
- 通过噪声图像的分组和前面一样（？）

维纳滤波器的系数通过基础分组的`3-D`变换中的能量进行定义
$$
\begin{equation}
\begin{split}
\boldsymbol W_{S_{x_R}^{wie}}=\frac{\mid \mathbb T_{3D}^{wie}\left( \widehat Y^{basic} \right) \mid^2}{\mid \mathbb T_{3D}^{wie}\left( \widehat Y^{basic} \right) \mid^2 + \sigma^2}
\end{split}
\tag{7}
\end{equation}
$$
然后进行维纳估计
$$
\begin{equation}
\begin{split}
\widehat Y^{wie}=\mathbb {T_{3D}^{wie}}^{-1}\left( \boldsymbol W_{S_{x_R}^{wie}} \mathbb T_{3D}^{wie}\left( \boldsymbol Z \right) \right)
\end{split}
\tag{8}
\end{equation}
$$
其中的$$Z$$是通过含噪声图像分组的结果。利用已经去除噪声的结果进行维纳系数的计算，然后将该滤波器应用于含噪声的图像中进行进一步的去噪。

#### B. 聚合全局估计

前面只讲了分组，后面说说估计中的**聚合**`Aggregation`这一步。通过变换域操作与反变换的估计方法，我们对图像的进行了**过度完整**的表达，这是因为前面的分块匹配中很容易出现重叠。另外我们很有可能在一个位置使用了多个估计。例如在不同参考下，某位置的块都能匹配到参考块，这样在两个位置进行处理的时候都会得到一个估计。当一个块匹配了许多参考的时候，他就会得到多个估计，而且这样的块不在少数。

不过好像一直没说怎么计算。由于一个位置可能出现较多的估计，我们采用加权平均的方式进行聚合。权重的选择比较重要。

- **权重**。通常逐块估计的结果是统计意义上相关的，有偏的，各个像素的方差也不同。然而，考虑到这些所有的因素就很困难了。我们发现对于方差较大的块赋予较小的权重，降低噪声的效果会好一些，即噪声越大的块应该赋予较小的权重。基于以上假设，如果我们的线性变换归一化，系数的平方代表功率，那么权重可以被赋予为

$$
\begin{equation}
\begin{split}
w_{x_R}^{ht}&=\frac{1}{\sigma^2 N_{har}^{x_R}}\\
w_{x_R}^{wie}&=\frac{1}{\sigma^2 \|W_{S^{wie}_{x_R}}\|_2^2}
\end{split}
\tag{9}
\end{equation}
$$

对于硬阈值后系数数量$$N$$小于1的情况，为了防止权重爆炸，这里直接置为$$1$$;由于块匹配计算的时候是采用重叠块方法的，需要考虑协方差（不过还是没有考虑），这里与样本方差成反比的结果不是完全成立的

- **加权平均聚合**。根据前面我们算出来的权重，来进行加权平均。我tm真是服了，你就一个加权平均都扯了一页多了，真的牛逼。重建图像的表达式为


$$
\begin{equation}
\begin{split}
\widehat y^{basic}(x)=\frac{\sum\limits_{x_R\in X}\sum\limits_{x_R\in S^{ht}_{x_R}}w^{ht}_{x_R}\widehat Y_{x_m}^{ht,x_R(x)}}{\sum\limits_{x_R\in X}\sum\limits_{x_R\in S^{ht}_{x_R}}w^{ht}_{x_R}\chi_{x_m}(x)}
\end{split}
\tag{10}
\end{equation}
$$
最终估计结果也可以这样计算，但是就要稍微变更一下表达式的形式。其中有一个函数$$\chi$$，是值域$$\{0,1\}$$的映射，若$$x_m$$在$$X$$内，则返回$$1$$，否则为$$0$$。

---

上面是论文的表述，实际上不是很容易理解。昨天我详细看了文章的代码，实际上`Aggregation`这一步就是做的简单加权和。由于考虑了边界效应加入了`Kaiser`窗函数，这里的流程其实就是，第一步聚合对于已经分组并且硬阈值的结果进行加权和，其权重估计考虑到了方差，加权是因为在每一个位置出现的像素块并不只利用了一次，逐块估计的时候，对于指定参考点的`Group`，实际上是离散余弦变换的集合。逆变换后乘我们估计的权值并送回他本来的位置

```python
basic_estimate[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup.shape[1],
                       BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup.shape[2]]
                       += BlockWeight * idct2D(BlockGroup[i, :, :])
```



### 实现（以及高效实现）

论文指出，如果直接实现，可能会比较慢。那么这里进行一点简化

- 滑动匹配的时候，采用每隔$$N_{sample}$$点进行一次采样的方式进行分块，这样可以显著降低处理数量。
- 将分组时数量设置上限，最多允许匹配$$N_2$$个块
- 搜索匹配分块的时候，只在以此位置为中心的$$N_S\times N_S$$个块。
- 这一步好傻逼不翻译了
- 将`3-D`变换拆成一个`2-D`和一个`1-D`的变换。
- 在处理之前预计算所有块的变换结果。
- 算法顺序可以进行适当调整
- 采用一个$$N_1 \times N_1$$大小的`Kaiser`窗（参数为$$\beta$$），来降低变换的边缘效应。



首先对于该算法，原始形式只能对灰度图像进行处理。因此对于彩色图像首先进行灰度化处理。为了便于后期处理。我们将图像转化为浮点型数据进行处理。灰度化之后的图像如下。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/GrayImage.png" style="display: inline-block;" width="500"/>
</div>


浮点化处理之后我们添加高斯白噪声。作为样例，我们设置白噪声功率为$$\sigma=25$$。添加噪声后的图像为

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/NoisyImage.png" style="display: inline-block;" width="500"/>
</div>


含噪的图像正式进入算法输入。首先进行第一步的基础估计。为了方便处理，我们先把后面可能会用到的离散余弦变换做掉。根据经验，这个过程大约是不到`5s`。

```
>>> DCT all in 4.547126293182373s
```

一般的，我们经验上会每隔三个点进行一次分块，那么原图$$512\times 512$$的大小就会被分割为$$170\times 170$$个`block`。分块并进行离散余弦变换后，开始分组。这里要进行离散余弦变换，是因为离散余弦变换结果具有很好的稀疏性，二维离散余弦变换后的结果往往会集中在左上角。我们这里举个例子，对于第一个`block`，分割结果为一个$$(16,8,8)$$大小的块，其中第一个变换结果是

```
[[ 1.29849999e+03  3.74591664e-01  1.83351953e+01 ...  3.02294573e+01
   -5.66313693e+01 -5.29934685e+00]
  [-2.50375759e+01  8.09275253e+00  7.49137009e+00 ... -1.07719190e+01
   -3.36913830e+01 -1.84984453e+01]
  [ 4.73723066e+01 -5.22545366e+00 -5.88699016e+00 ... -3.96155164e+01
   -1.19085326e-01  4.44470636e+01]
  ...
  [ 8.13641550e+00  2.00919075e+01  3.89090860e+00 ...  3.27180244e+01
   -3.36336860e+00  1.20396360e+01]
  [-3.00899028e+01 -7.98224680e+01  9.95073699e+00 ... -4.43562755e+01
    2.96699942e+01  2.87377413e+01]
  [ 5.67538581e+01 -3.84358363e+01 -2.19652484e+01 ...  3.70353163e+00
    1.51453117e+01 -2.34829764e+01]]
```

这是一个矩阵的形式，可以看出主要的能量都集中在了左上角。

由于变换结果都是稀疏的，所以都是一个以左上角为峰的结构，所以进行距离比较的时候，结果匹配会比直接进行图像相关匹配效果更好。

在程序中，我们直接将分块结果用离散余弦变换来保存了，这将有利于在后面的一步应用拆分的`3-D`变换。

分组后我们直接进行`3-D`滤波。由于`Group`在建立的时候都是采用的离散余弦变换，因此我们得到的直接就是已经完成了`2-D`变换的分组。接下来要实现`3-D`变换，只需要在没有变换过的轴上应用变换即可。此时对于有最大$$16$$个元素的轴，我们进行一维的`dct`，将得到一个主要成分集中在前几个元素的变换结果。通过硬阈值处理这个结果，我们相当于逐行处理整个`Group`的`3-D`变换结果。

进行了硬阈值处理后（注意这里要记录阈值处理后的非零点数量）我们进行反变换，首先是对于刚刚处理的那个维度的反变换，然后是两个之前做`2-D dct`的维度。但是这里仍然不进行反变换，后面还会用到。

给一个例子，对于刚才的那个矩阵的第一行变换，结果是这样的

```
[5007.15471506    0.            0.            0.            0.
    0.           79.37501785    0.            0.            0.
    0.            0.            0.            0.            0.
    0.        ]
```

可见非零点数很少，整个矩阵十分稀疏。

完成了`3-D`滤波后我们进行第一次聚合重建。由于第一次的主要处理就是阈值，这里的聚合重建主要就是加权平均与反变换的过程。加权平均采用了$$8\times 8$$`Kaiser`窗函数，这个窗函数是加在反变换后的结果上的，不过随着图像的重建过程，这个窗函数只是一个初值的意义，据论文的意思是可以降低边缘效应。

注意这里的重建过程，在以每一个参考点为基准时，我们都会至多同时恢复$$16$$个位置的图像，每次的恢复都将是一个占一定权重的部分，这个权重是视噪声而定的。我们给出一张当过程进行到中途（这里是$$80\%$$）的时候重建的图像。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/BasicEsti80.png" style="display: inline-block;" width="500"/>
</div>

可见图像是自左上到右下逐个`Block Group`重建的。

这里给出第一步估计的结果，如图所示

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/BasicEsti.png" style="display: inline-block;" width="500"/>
</div>

可以看出的不同点是，基础重建的图像在细节上少了一些，例如帽子上的纹理，但是头发这些细节恢复地较好，可能是存在较好的匹配样本。说白了图像重建算法，就是根据现有的信息去估计原有的信息，避免不了各种位置的求均值算法。最后的模糊也是因为均值过程相当于一个低通滤波的效果，去除了图像原有的细节。保留细节是一个挺困难的事情，而深度学习方法可以通过算法本身的特点对细节进行一定的增强，至于效果如何我们不能直接下定论。

接下来使用这个结果进行第二步的估计。第二步的许多操作与第一步类似。分组这里，实际上是重新分组，在流程图中也表现出了分组的明显不同。由于第一步已经给出了基础估计，此时假设这个就是对原图的无偏估计了，我们利用这个图进行重新分组，并且将这个分组方案直接应用于含噪图像。

为了处理方便，这里还是预先计算了两个图像的离散余弦变换，并且同样通过迭代的方法进行逐块重建。与上一次不同的是，这里不再通过硬阈值方法进行处理，而是采用维纳滤波的方法。

维纳滤波器是特指**最小化均方误差**得到的滤波器。而为了最小化均方误差，一般是需要一定的先验知识的，例如一维信号的维纳滤波器形式
$$
\begin{equation}
\begin{split}
\boldsymbol w=\boldsymbol R^{-1}\boldsymbol p
\end{split}
\tag{11}
\end{equation}
$$
其中$$R$$是输入信号的自相关矩阵，$$p$$是输入信号与目标信号的互相关矩阵。这里的互相关计算就是需要一定的先验知识的，因为目标信号往往是我们需要求解的信号，我们并不能得到这个信号，只能估计。这里对真实结果的估计采用的就是第一步中计算的结果。经过维纳滤波后的结果进行反变换就得到了最终估计图像。

*实话讲吧，这个结果我感觉还不如基础估计得到的。*

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/BasicEsti.png" style="display: inline-block;" width="250"/>
<img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/FinalEsti.png" style="display: inline-block;" width="250"/>
</div>

### 总结

基础估计信噪比 27.46 dB

最终估计信噪比 27.10 dB

原来老师讲的东西很多都是会用得上的。大三上学期老师说的常见降噪算法中利用硬阈值算法和软阈值算法这里就遇到了，利用小波域稀疏性的特点这里也用到了。

图像处理的降噪过程是困难的，我们没有对图像的先验知识的时候，或者先验知识信息量较少难以利用的时候，对图像的降噪处理就是一个瞎编的过程。我们一开始学习了基于各类简单噪声的去除，又稍微讲解了一下如何对图像的退化做复原。再往后，我们学了基于对整体方差的估计方法做去噪处理，对局部方差的估计，到现在这种复杂的算法，实际上去噪就是离不开不断的加权平均，不停地估计。但是手头能够利用的数据就只有那一张图像，甚至都没有噪声的信息，在实际应用中我们又能获得怎么样的效果呢？




## BM3D 算法的复数域推广

这里还没做完，计划是先做一个简单的一维信号重建。



## 代码

这里给出一维实现

```python
# 2019-5-21 20:12:16
# Personal implementation of BM3D Denoising Algorithm

import numpy as np
import time
import cv2
from scipy.fftpack import dct, idct
from matplotlib import pyplot as plt
__author__ = 'Prophet'

# Parameters

sigma = 25  # add-noise var
S1_blocksize = 8
S1_Threshold = 5000
S1_MaxMatch = 16
S1_DownScale = 1
S1_WindowSize = 50
lamb2d = 2.0
lamb3d = 2.7

S2_Threshold = 400
S2_MaxMatch = 32
S2_blocksize = 8
S2_DownScale = 1
S2_WindowSize = 50

beta = 2.0
A = 255

# Utils


def signalGen(A=255, f=10, fs=512, show=1):
    '''
    Sine Signal Demo
    '''
    t = np.linspace(0, 2 * np.pi * f, fs)
    if show:
        plt.plot(A * np.sin(t))
        plt.show()
    return A * np.sin(t)


def AWGN(sigma, signal, show=1):
    '''
    Add White Gauss Noise
    '''
    noise = sigma * np.random.randn(len(signal))
    if show:
        plt.plot(noise + signal)
        plt.show()
    return signal + noise, noise


def cvInit():
    print('BM3D Initializing at {}'.format(time.time()))
    print('Initializing OpenCV')
    start = time.time()
    cv2.setUseOptimized(True)
    end = time.time()
    print('Initialized in {}s'.format(end - start))
    pass


def preDCT(Img, blocksize, timer=True):
    s = time.time()
    BlockDCT_all = np.zeros((Img.shape[0] - blocksize, Img.shape[1] - blocksize, blocksize, blocksize),
                            dtype=float)
    for i in range(BlockDCT_all.shape[0]):
        for j in range(BlockDCT_all.shape[1]):
            Block = Img[i:i + blocksize, j:j + blocksize]
            BlockDCT_all[i, j, :, :] = dct2D(Block.astype(np.float64))
    if timer:
        print('DCT all in {}s'.format(time.time() - s), end='\n')
    return BlockDCT_all


def preDCT1D(noisySignal, blocksize, timer=True):
    s = time.time()
    blockDCT1D_all = np.zeros(
        (noisySignal.shape[0] - blocksize, blocksize), dtype=float)
    for i in range(blockDCT1D_all.shape[0]):
        Block = noisySignal[i:i + blocksize]
        blockDCT1D_all[i, :] = dct(Block.astype(np.float64))
    if timer:
        print('DCT all in {}s'.format(time.time() - s), end='\n')
    return blockDCT1D_all
    pass


def searchWindow1D(sig, RefPoint, blocksize, WindowSize):
    '''
        Set Boundary
    '''
    if blocksize >= WindowSize:
        print('Error: blocksize is smaller than WindowSize.\n')
        exit()
    Margin = np.zeros((2, 1), dtype=int)
    Margin[0, 0] = max(
        0, RefPoint + int((blocksize - WindowSize) / 2))  # left-top x
    Margin[1, 0] = Margin[0, 0] + WindowSize  # right-bottom x
    if Margin[1, 0] >= sig.shape[0]:
        Margin[1, 0] = sig.shape[0] - 1
        Margin[0, 0] = Margin[1, 0] - WindowSize

    return Margin


def computeSNR(signal, noise, estimate=None):
    if estimate is None:
        return 10 * np.log10(np.var(signal) / np.var(noise))
    else:
        return 10 * np.log10(np.var(signal) / np.var(estimate - signal))


# ===========================================================================


def S1_ComputeDist1D(BlockDCT1, BlockDCT2):
    """
    Compute the distance of two DCT arrays *BlockDCT1* and *BlockDCT2*
    """
    if BlockDCT1.shape != BlockDCT2.shape:
        print(
            'ERROR: two DCT Blocks are not at the same shape in step1 computing distance.\n')
        exit()
    blocksize = BlockDCT1.shape[0]
    if sigma > 40:
        ThreValue = lamb2d * sigma
        BlockDCT1 = np.where(abs(BlockDCT1) < ThreValue, 0, BlockDCT1)
        BlockDCT2 = np.where(abs(BlockDCT2) < ThreValue, 0, BlockDCT2)
    return np.linalg.norm(BlockDCT1 - BlockDCT2)**2 / (blocksize**2)


def S1_Grouping1D(noisyImg, RefPoint, blockDCT1D_all, blocksize, ThreDist, MaxMatch, WindowSize):
    # initialization, get search boundary
    WindowLoc = searchWindow1D(noisyImg, RefPoint, blocksize, WindowSize)
    # print(WindowLoc)
    # number of searched blocks
    Block_Num_Searched = (WindowSize - blocksize + 1)
    # print(Block_Num_Searched)
    # 0 padding init
    BlockPos = np.zeros((Block_Num_Searched, 1), dtype=int)
    BlockGroup = np.zeros(
        (Block_Num_Searched, blocksize), dtype=float)
    Dist = np.zeros(Block_Num_Searched, dtype=float)

    RefDCT = blockDCT1D_all[RefPoint, :]
    # print(RefDCT)
    match_cnt = 0
    # k = []
    for i in range(WindowSize - blocksize + 1):
        # for j in range(WindowSize - blocksize + 1):
        SearchedDCT = blockDCT1D_all[WindowLoc[0, 0] + i, :]
        dist = S1_ComputeDist1D(RefDCT, SearchedDCT)
        # k.append(dist)
        if dist < ThreDist:
            BlockPos[match_cnt, :] = [
                WindowLoc[0, 0] + i, ]
            # Block Group has DCT data inside.
            # convenient for 3-D Transform
            BlockGroup[match_cnt, :] = SearchedDCT
            Dist[match_cnt] = dist
            match_cnt += 1
    # print(k)
    # print(sorted(k))
    '''
    if match_cnt == 1:
        print('WARNING: no similar blocks founded for the reference block {} in basic estimate.\n'\
              .format(RefPoint))
    '''
    if match_cnt <= MaxMatch:
        # less than MaxMatch similar blocks founded, return all similar blocks
        BlockPos = BlockPos[:match_cnt, :]
        BlockGroup = BlockGroup[:match_cnt, :]
    else:
        # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks
        # These lines are awesome.
        idx = np.argpartition(Dist[:match_cnt], MaxMatch)
        BlockPos = BlockPos[idx[:MaxMatch], :]
        BlockGroup = BlockGroup[idx[:MaxMatch], :]
    return BlockPos, BlockGroup


def S1_2DFiltering(BlockGroup):
    # print(BlockGroup)
    ThreValue = lamb3d * sigma * 8
    nonzero_cnt = 0
    # since 2D transform has been done, we do 1D transform, hard-thresholding and inverse 1D
    # transform, the inverse 2D transform is left in aggregation processing

    for i in range(BlockGroup.shape[1]):  # shape=(16,8)
        # print(BlockGroup[:, i, j])
        ThirdVector = dct(BlockGroup[:, i], norm='ortho')  # 1D DCT
        # print(abs(ThirdVector[:]) < ThreValue)
        ThirdVector[abs(ThirdVector[:]) < ThreValue] = 0.
        # print(ThirdVector)
        # exit()
        nonzero_cnt += np.nonzero(ThirdVector)[0].size
        BlockGroup[:, i] = list(idct(ThirdVector, norm='ortho'))
    return BlockGroup, nonzero_cnt
    pass


def S1_Aggregation(BlockGroup, BlockPos, basic_estimate, basicWeight, basicKaiser, nonzero_cnt):

    if nonzero_cnt < 1:
        BlockWeight = 1.0 * basicKaiser
    else:
        BlockWeight = (1. / (sigma ** 2 * nonzero_cnt)) * basicKaiser
    # print(BlockPos.shape[0])
    for i in range(BlockPos.shape[0]):
        # print(BlockPos[i, 0], BlockPos[i, 0] + BlockGroup.shape[1])
        basic_estimate[BlockPos[i, 0]:BlockPos[i, 0] +
                       BlockGroup.shape[1]] += np.dot(BlockWeight, np.diag(idct(BlockGroup[i, :])))
        basicWeight[BlockPos[i, 0]:BlockPos[i, 0] +
                    BlockGroup.shape[1]] += BlockWeight


def BM3D1D_S1(noisySignal, para=None):
    # Using global variables
    basicEstimate = np.zeros(noisySignal.shape, dtype=float)
    basicWeight = np.zeros(noisySignal.shape, dtype=float)
    window = np.array(np.kaiser(S1_blocksize, beta))
    # basicKaiser = np.array(window.T * window)
    basicKaiser = window

    blockDCT1D_all = preDCT1D(noisySignal, S1_blocksize)

    all_ = int((noisySignal.shape[0] - S1_blocksize) / S1_DownScale) + 2
    print('{} iterations remain.'.format(all_))
    count = 0
    start_ = time.time()

    for i in range(all_):
        if i != 0:
            print('i={}; Processing {}% ({}/{}), consuming {} s'.format(i,
                                                                        count * 100 / all_, count, all_, time.time() - start_))
        count += 1
        RefPoint = min(S1_DownScale * i,
                       noisySignal.shape[0] - S1_blocksize - 1)
        BlockPos, BlockGroup = S1_Grouping1D(
            noisySignal, RefPoint, blockDCT1D_all, S1_blocksize, S1_Threshold, S1_MaxMatch, S1_WindowSize)
        # print(BlockPos, BlockGroup)
        # exit()
        BlockGroup, nonzero_cnt = S1_2DFiltering(BlockGroup)
        S1_Aggregation(BlockGroup, BlockPos, basicEstimate,
                       basicWeight, basicKaiser, nonzero_cnt)
    basicWeight = np.where(basicWeight == 0, 1, basicWeight)
    basicEstimate[:] /= basicWeight[:]
    return basicEstimate


# ==========================================================

def S2_Aggregation1D(BlockGroup_noisy, WienerWeight, BlockPos, finalImg, finalWeight, finalKaiser):
    """
    Compute the final estimate of the true-image by aggregating all of the obtained local estimates
    using a weighted average
    """
    BlockWeight = WienerWeight * finalKaiser
    for i in range(BlockPos.shape[0]):
        finalImg[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup_noisy.shape[1]
                 ] += BlockWeight * idct(BlockGroup_noisy[i, :])
        finalWeight[BlockPos[i, 0]:BlockPos[i, 0] +
                    BlockGroup_noisy.shape[1]] += BlockWeight


def S2_2DFiltering(BlockGroup_basic, BlockGroup_noisy):
    """
    Wiener Filtering
    """
    Weight = 0
    coef = 1.0 / BlockGroup_noisy.shape[0]
    for i in range(BlockGroup_noisy.shape[1]):
        # for j in range(BlockGroup_noisy.shape[2]):
        Vec_basic = dct(BlockGroup_basic[:, i], norm='ortho')
        Vec_noisy = dct(BlockGroup_noisy[:, i], norm='ortho')
        Vec_value = Vec_basic**2 * coef
        Vec_value /= (Vec_value + sigma**2)  # pixel weight
        Vec_noisy *= Vec_value
        Weight += np.sum(Vec_value)
        '''for k in range(BlockGroup_noisy.shape[0]):
                Value = Vec_basic[k]**2 * coef
                Value /= (Value + sigma**2) # pixel weight
                Vec_noisy[k] = Vec_noisy[k] * Value
                Weight += Value'''
        BlockGroup_noisy[:, i] = list(idct(Vec_noisy, norm='ortho'))
    if Weight > 0:
        WienerWeight = 1. / (sigma**2 * Weight)
    else:
        WienerWeight = 1.0
    return BlockGroup_noisy, WienerWeight


def S2_ComputeDist1D(sig, Point1, Point2, BlockSize):
    """
    Compute distance between blocks whose left-top margins' coordinates are *Point1* and *Point2*
    """
    Block1 = (sig[Point1[0]:Point1[0] + BlockSize]).astype(np.float64)
    Block2 = (sig[Point2[0]:Point2[0] + BlockSize]).astype(np.float64)
    return np.linalg.norm(Block1 - Block2)**2 / (BlockSize**2)


def S2_Grouping1D(basicEstimate, noisyImg, RefPoint, BlockSize, ThreDist, MaxMatch, WindowSize,
                  BlockDCT_basic, BlockDCT_noisy):
    WindowLoc = searchWindow1D(basicEstimate, RefPoint, BlockSize, WindowSize)

    Block_Num_Searched = (WindowSize - BlockSize + 1)

    BlockPos = np.zeros((Block_Num_Searched, 1), dtype=int)
    BlockGroup_basic = np.zeros(
        (Block_Num_Searched, BlockSize), dtype=float)
    BlockGroup_noisy = np.zeros(
        (Block_Num_Searched, BlockSize), dtype=float)
    Dist = np.zeros(Block_Num_Searched, dtype=float)
    match_cnt = 0
    for i in range(WindowSize - BlockSize + 1):
        SearchedPoint = [WindowLoc[0, 0] + i]
        dist = S2_ComputeDist1D(
            basicEstimate, [RefPoint], SearchedPoint, BlockSize)
        if dist < ThreDist:
            BlockPos[match_cnt, :] = SearchedPoint
            Dist[match_cnt] = dist
            match_cnt += 1
    if match_cnt <= MaxMatch:
        BlockPos = BlockPos[:match_cnt, :]
    else:
        idx = np.argpartition(Dist[:match_cnt], MaxMatch)
        BlockPos = BlockPos[idx[:MaxMatch], :]
    for i in range(BlockPos.shape[0]):
        SimilarPoint = BlockPos[i, :]
        BlockGroup_basic[i, :] = BlockDCT_basic[SimilarPoint[0], :]
        BlockGroup_noisy[i, :] = BlockDCT_noisy[SimilarPoint[0], :]
    BlockGroup_basic = BlockGroup_basic[:BlockPos.shape[0], :]
    BlockGroup_noisy = BlockGroup_noisy[:BlockPos.shape[0], :]
    return BlockPos, BlockGroup_basic, BlockGroup_noisy


def BM3D1D_S2(noisySignal, basicEstimate):
    finalEstimate = np.zeros(noisySignal.shape, dtype=float)
    finalWeight = np.zeros(noisySignal.shape, dtype=float)
    Window = np.array(np.kaiser(S2_blocksize, beta))
    finalKaiser = Window

    BlockDCT_noisy = preDCT1D(noisySignal, S2_blocksize)
    BlockDCT_basic = preDCT1D(basicEstimate, S2_blocksize)
    count = 0
    start_ = time.time()
    all_ = int((basicEstimate.shape[0] - S2_blocksize) / S2_DownScale) + 2

    for i in range(int((basicEstimate.shape[0] - S2_blocksize) / S2_DownScale) + 2):
        print('i={}; Processing {}% ({}/{}), consuming {} s'.format(i,
                                                                    count * 100 / all_, count, all_, time.time() - start_))
        RefPoint = min(S2_DownScale * i,
                       basicEstimate.shape[0] - S2_blocksize - 1)
        BlockPos, BlockGroup_basic, BlockGroup_noisy = S2_Grouping1D(basicEstimate, noisySignal,
                                                                     RefPoint, S2_blocksize,
                                                                     S2_Threshold, S2_MaxMatch,
                                                                     S2_WindowSize,
                                                                     BlockDCT_basic,
                                                                     BlockDCT_noisy)

        BlockGroup_noisy, WienerWeight = S2_2DFiltering(
            BlockGroup_basic, BlockGroup_noisy)
        S2_Aggregation1D(BlockGroup_noisy, WienerWeight, BlockPos, finalEstimate, finalWeight,
                         finalKaiser)
        count += 1

    finalWeight = np.where(finalWeight == 0, 1, finalWeight)
    finalEstimate[:] /= finalWeight[:]
    return finalEstimate
# ==========================================================


if __name__ == '__main__':
    cvInit()

    sig = signalGen(A)
    noisySignal, noise = AWGN(25, sig)
    print(np.var(noisySignal))

    S1_start = time.time()
    signalBasicEstimate = BM3D1D_S1(noisySignal) / np.sqrt(A)
    print('\nFinish Basic Estimate in {} s'.format(time.time() - S1_start))
    plt.plot(signalBasicEstimate)
    plt.show()

    print('signal power:\t{}'.format(np.var(sig)))
    print('noise power:\t{}'.format(np.var(noise)))
    print('SNR:\t\t\t{} dB'.format(computeSNR(sig, noise)))
    print('basic estimate SNR:\t{} dB'.format(
        computeSNR(sig, None, signalBasicEstimate)))

    S2_start = time.time()
    finalEstimate = BM3D1D_S2(signalBasicEstimate, noisySignal) / np.sqrt(A)
    print('\nFinish Final Estimate in {} s'.format(time.time() - S2_start))
    plt.plot(finalEstimate)
    plt.show()

    print('signal power:\t{}'.format(np.var(sig)))
    print('noise power:\t{}'.format(np.var(noise)))
    print('SNR:\t\t\t{} dB'.format(computeSNR(sig, noise)))
    print('final estimate SNR:\t{} dB'.format(
        computeSNR(sig, None, finalEstimate)))

    exit()
    # Basic_PSNR = computePSNR()

```








