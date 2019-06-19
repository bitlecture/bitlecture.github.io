---
layout: post
comments: true
title:  "Unsupervised learning based fast beamforming design for downlink MIMO"
excerpt: "-"
date:   2019-04-03 12:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
---


# Unsupervised learning based fast beamforming design for downlink MIMO

一种基于无监督学习方法的`MIMO`下行链路快速波束成形方法

## Abstract

在下行链路传输场景中，若使用多天线阵列，在发射机上的功率分配和波束成形设计是十分重要的。本文考虑了一个`MIMO`广播信道在系统功率约束条件下来最大化加权`sum-rate`。传统的加权最小化均方误差算法`WMMSE`可能会获得次优解，但是涉及较高的计算复杂度。为了降低这个复杂度，我们提出了一个基于非监督学习的快速波束成形设计方法，该方法离线训练深度神经网络并实时在线提供服务。训练过程是基于一种端到端方法，不需要人为标记样本，从而避免了打标的复杂过程。此外我们使用了基于`APoZ`剪枝算法来压缩神经网络的容量，从而可以部署到低运算容量的设备中。最后的实验结果显示这个方法显著提高了运算速度，而性能表现接近`WMMSE`算法。

## INDEX TERMS

多入多出，波束成形，深度学习，无监督学习，网络剪枝

## I. Introduction

在数据流量的高速增长背景下，下一代无线通信系统需要提供更大的吞吐量来达到更到的更高的数据速率需求。多入多出`MIMO`技术被认为是是通过增加收发器处的天线数量来利用空间资源的有效方式。因此，如果我们知道准确的信道状态信息`CSI`，则`MIMO`可以通过使用线性或非线性传输技术显著地改善信道容量。对于一个`MIMO`广播下行链路场景而言，非线性传输技术中的`DPC(Dirty Paper Coding)`技术可以达到下行链路信道容量的理论上界。然而由于这个技术需要大量的计算量，这个技术从理论到实现还有一定距离。因此线性下行链路传输技术（也被称为波束成形技术）由于其设计简单运算复杂度小被广泛采用。

一个关于`Weighted Sum-Rate (WSR)`最大化问题的热门的算法就是`WMMSE`算法。`WSR`最大化问题可以转化为一个`WMMSE`最大化问题，通过迭代更新权重矩阵来设计波束成形方法。然而`WMMSE`的计算复杂度随着变量的数增加而增加，由于`WMMSE`算法中含有许多复杂的运算，比如每次迭代时需要进行矩阵求逆等。另一种波束成形设计方法结合了`zero-force`和`water-fill`算法。在有干扰的`MIMO`系统中使用的`water-fill`算法中，每次迭代中的奇异值分解操作也会消耗大量的计算资源，这可能导致较大的延迟。这些基于严格数学模型的传统算法可以获得令人满意的性能，但是由于较高的计算复杂度导致的严重延迟，可能导致它们无法满足实时性需求。事实上低延迟与低功耗需求在下一代无线通信技术中是普遍存在的。例如载具之间的通信只能容忍几毫秒的延迟；而物联网中的无线传感器则要求很低的能量消耗。

*吹B开始*

随着深度学习的发展，神经网络算法在无线通信领域由于其强大的特征提取与展现的能力而受到了广泛关注。深度学习辅助技术实现离线学习与在线部署训练好的网络，和迭代算法相比将大大降低时间复杂度。因为训练后的神经网络只包含简单的线性和非线性转换单元，这些操作的计算复杂度极低而且有相当不错的性能表现。在功率控制问题中，深度神经网络可以被用作一个高性能拟合器来接近`WMMSE`算法的性能表现。深度学习中的自编码器广泛应用于非正交多址`NOMA`通信系统，在优化通信系统性能的同时实现了端到端通信的新机制。`CsiNet`是使用新颖的`CSI`感知与重建机制开发的。它更有效地探索了信道的结构信息，提高了系统的计算效率。深度学习技术也是毫米波通信中的一个热门研究课题，由于毫米波系统的复杂性，上述数据驱动的深度学习技术并不能直接应用。因此，模型驱动的深度学习技术（最先在图像处理领域中兴起的）首先应用于毫米波通信系统的可解释性。一些人已经提出了一种模型驱动的深度网络结构，它结合了深度学习和传统算法，用于毫米波`Massive MIMO`中的信道估计。通过分解传统算法并将其中的某些部分用深度学习方法代替（例如卷积神经网络`CNN`），传统算法的性能就被显著提升了。降低算法的迭代次数显著降低了计算法复杂性。为了继续加速神经网络运算并降低内存使用率，轻量网络成为了一个研究的重点课题。这篇文章探索了神经网络的加速运算技术，通过在轻量网络中应用网络剪枝技术来进行波束成形设计。

在文献中，先前的工作主要是考虑到功率控制与单天线收发机的通信场景。一个与先前的波束成形工作的主要不同点在于，我们的优化目标是功率约束条件下的最大化`sum-rate`。在这项研究中，我们测试了一种将深度神经网络应用在功率约束的`MIMO`系统中最大化`sum-rate`的全新结构。本文的主要贡献是

- 在`MIMO`下行链路场景中，采用了基于深度神经网络的方案进行波束成形设计。深度神经网络被用于获取信道的结构信息，其中深度神经网络可以被看作是一个有着多重输入输出的多层网络结构的黑盒子，用来实现端到端波束成形设计。

- 通过重新设计损失函数，提出了一种基于深度神经网络的波束成形设计结构。基于非监督学习方法的思想，`sum-rate`可以在功率约束的条件下最大化，其性能表现只比`WMMSE`差一点。
- 为了进一步计算网络的运算，并降低内存使用，本文使用剪枝算法探索了网络加速方法，并尝试应用。通过剪枝算法，网络的参数压缩了，这可以进一步降低网络的运算复杂度。

本文剩余部分是如下组织的。在第`II`部分，我们描述了系统模型和问题。第`III`部分我们提出了深度神经网络的方案与学习策略。仿真结果在第`IV`部分，最后是结论总结。

## II. System Model and Problem Formulation

### A. System Model

首先我们考虑一个`MIMO`传输系统的典型下行链路场景。如图所示，一个收发机有$$P$$个天线、服务$$K$$个用户。这些用户每人都有$$Q$$个接收天线。用户$$k$$与广播卫星（原文**BS**，目测`Broadcast Satellite`）之间的信道可以描述为$$\boldsymbol{H}_k \in \mathbb{C}^{[Q\times P]}$$，这个是信道矩阵，包含了不同收发机天线对之间的增益。用户接收信号可以表示为
$$
\begin{equation}
\begin{split}
\boldsymbol y_k =\boldsymbol H_k\boldsymbol s+\boldsymbol n_k
\end{split}
\tag{1}
\end{equation}
$$

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/0403fig1.png" style="display: inline-block;" width="500"/>
</div>

这里$$\boldsymbol{s} \in \mathbb{C}^{[P\times 1]}​$$代表了发射向量，$$\boldsymbol{n}_k \in \mathbb{C}^{[Q\times 1]}​$$代表噪声向量。不同用户之间的协方差都是$$\sigma^2\boldsymbol{I}_Q​$$。发射向量$$\boldsymbol{s}​$$可以被表示为经过$$K​$$阶滤波器的数据向量$$\boldsymbol{x}_i​$$

$$
\begin{equation}
\begin{split}
\boldsymbol s =\sum_{k=1}^{K}\boldsymbol W_k\boldsymbol x_k
\end{split}
\tag{2}
\end{equation}
$$
其中矩阵$$\boldsymbol W_k$$是线性发射波束成型器，$$\boldsymbol x_k$$是输入向量。我们假设每个用户独立接收数据流。发射向量$$\boldsymbol s$$应当满足的块功率约束为
$$
\begin{equation}
\begin{split}
\mathbb{E}[\boldsymbol s_k^H\boldsymbol s_k]=\sum_k Tr(\boldsymbol W_k \boldsymbol W_k^H)\leq p_{max}
\end{split}
\tag{3}
\end{equation}
$$
同样假定发射机处有着无延迟的`CSI`，且信道矩阵在传输过程中是一个常量矩阵。

### B. Problem Formulation

我们的主要目标是最大化所有用户的`WSR`，通过设计线性传输滤波器$$\boldsymbol W_k​$$。最大化问题就可以写作是
$$
\begin{equation}
\begin{split}
[\boldsymbol W_1,...,\boldsymbol W_k]=&argmax \sum_k u_kR_k\\
&s.t. \sum_{k=1}^K Tr(\boldsymbol W_k \boldsymbol W_k^H)\leq p_{max}
\end{split}
\tag{4}
\end{equation}
$$
此时我们定义$$R_k$$是用户$$k$$的速率，而$$u_k$$是他的权重。此处使用高斯分布的信号作为测试信号，因此可以达到的最高速率可以表示为
$$
\begin{equation}
\begin{split}
R_k=\log \det(\boldsymbol I_k + \boldsymbol W_k^H\boldsymbol W_k^H\boldsymbol J_{\widetilde{v}_k\widetilde{v}_k}^{-1}\boldsymbol H_k\boldsymbol W_k)
\end{split}
\tag{5}
\end{equation}
$$
其中$$\boldsymbol J$$代表接收器$$\widetilde{v}$$上的有效噪声和干扰协方差矩阵
$$
\begin{equation}
\begin{split}
\boldsymbol J_{\widetilde{v}_k\widetilde{v}_k}=\boldsymbol I_k+\sum_{i=1,i\neq k}^{K} \boldsymbol H_k \boldsymbol W_i \boldsymbol W_i^H \boldsymbol H_k^H
\end{split}
\tag{6}
\end{equation}
$$
另外我们定义$$\boldsymbol W_{[P\times QK]}=[\boldsymbol W_1,...,\boldsymbol W_k]$$，$$\boldsymbol H_{[QK\times P]}=[\boldsymbol H_1^H,...,\boldsymbol H_k^H]$$，这两个是块矩阵，包含的是每一个用户的发射成形滤波器和信道增益。

## III. Proposed Method

这一部分我们使用深度学习结构来设计波束成形。我们首先简单描述深度神经网络的结构，然后提出了两个学习方法，即监督学习法和非监督学习法。我们还解释了如何在训练过程中同时实现主要目标和约束条件。最后我们介绍如何对网络进行剪枝，从而降低复杂性。

### A. Proposed Deep Neural Network Architecture

这一段简单写写得了。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/0403fig2.png" style="display: inline-block;" width="800"/>
</div>
我就直接说了，这个网络结构真的是简单的不行，基本上没有什么需要说明的。就是通过用户的信道矩阵输入，通过全连接层传递和`LeakyReLU+Dropout`层进行激活和剪枝，最终输出估计结果，然后根据功率限制要求输出一个期望的结果。中间的隐藏层神经元数量分别为$$200,300,200$$，三层。由于考虑到可能存在负值，这里没有直接采用`ReLU`，而是`LeakyReLU`作为激活函数。

### B. Learning Policy

这一部分介绍非监督学习训练策略。为了凸显非监督学习的优越性，监督学习的方式也会简单介绍一下。

#### Supervised Learning(DNN-Supervised)

监督学习的工作方式就像是一个函数拟合器，主要是训练神经网络尽可能去近似`WMMSE`的正确结果。监督学习的样本集为$$\Gamma$$，则训练数据表示为$$(\boldsymbol H^{(i)},\boldsymbol W^{(i)})_{i\in\Gamma}$$，即信道矩阵和波束成形阵。在数据初始化的阶段，我们采用发送匹配滤波器去初始化。使用一个常系数去保证功率能够满足限制。我们选择的损失函数是均方误差损失函数。采用`Adam`优化器，来保证即使问题是非凸优化问题也能正常解决。

#### UNsupervised Learning(DNN-Unsupervised)

监督学习问题中的输出是我们采用`WMMSE`计算的正确结果，输入输出是相互一一对应的；非监督学习问题则不同，这里的训练是不需要标记的。我们构造一个如下的损失函数
$$
\begin{equation}
\begin{split}
\boldsymbol{l}(\boldsymbol{\theta};\boldsymbol{h};\boldsymbol{\hat w})=&-\sum_{k=1}^K \log \det(\boldsymbol I_k + \boldsymbol {\widehat W}_k^H\boldsymbol W_k^H\boldsymbol {\widehat J}_{\widetilde{v}_k\widetilde{v}_k}^{-1}\boldsymbol H_k\boldsymbol {\widehat W}_k)\\=&-\sum_{k=1}^K\widehat{R}_k
\end{split}
\tag{7}
\end{equation}
$$
这里$$\boldsymbol{\theta},\boldsymbol{\hat w}​$$分别是深度神经网络的参数和输出。$$\boldsymbol {\widehat J}_{\widetilde{v}_k\widetilde{v}_k}=\boldsymbol I_k+\sum_{i=1,i\neq k}^{K} \boldsymbol H_k \boldsymbol {\widehat W}_i \boldsymbol {\widehat W}_i^H \boldsymbol H_k^H​$$代表的是对原来准确值的估计。考虑到约束条件，我们重写损失函数，增加一个惩罚项
$$
\begin{equation}
\begin{split}
\mathfrak{L}(\boldsymbol{\theta};\boldsymbol{h};\boldsymbol{\hat w})=\boldsymbol{l}(\boldsymbol{\theta};\boldsymbol{h};\boldsymbol{\hat w})+\lambda \mid \Omega(\widehat{w}) \mid
\end{split}
\tag{8}
\end{equation}
$$
我们说$$\lambda​$$是一个超参数，需要谨慎选择；$$\Omega(\widehat w)=\sum_{k=1}^K Tr(\boldsymbol {\widehat W}_i \boldsymbol {\widehat W}_i^H)<p_{max}​$$。然后深度神经网络需要解决的问题就变成了
$$
\begin{equation}
\begin{split}
\boldsymbol \theta^*=\underset{\boldsymbol \theta}{argmin}\mathfrak{L}(\boldsymbol{\theta};\boldsymbol{h};\boldsymbol{\hat w})
\end{split}
\tag{9}
\end{equation}
$$
略微进行一点变形可以得到
$$
\begin{equation}
\begin{split}
\boldsymbol \theta^*=\underset{\boldsymbol \theta}{argmin}\mathfrak{L}(\boldsymbol{\theta};\boldsymbol{h};NET(\boldsymbol \theta,\boldsymbol h))
\end{split}
\tag{10}
\end{equation}
$$
这样约束条件就转化成了一个函数优化问题，这个形式与正则化训练相同（这里可以看前面的机器学习中的介绍，正则化是为了降低高次项的影响，突出强调其他变量的重要性的）。这里就可以采用反向传播和`Adam`优化器了。

那么无论是使用了监督学习还是非监督学习，最终我们都要再进行一个归一化操作，即
$$
\begin{equation}
\begin{split}
\boldsymbol {\widehat W^{DNN}}=b\boldsymbol {\widehat W}
\end{split}
\tag{10}
\end{equation}
$$
其中$$b=\sqrt{\frac{p_{max}}{Tr(\boldsymbol {\widehat W} \boldsymbol {\widehat W}^H)}}$$，这样就可以将功率规定化到指定的约束条件。

这个网络中首先输入的是信道矩阵，通过信道矩阵估计出成形滤波器的取值，然后通过这两个结果计算出有效噪声与干扰，并可以根据这个结果计算出速率。我们要让速率最大化，那么就让负速率最小化。后面再加一个功率约束条件，然后搜索在这个情况下的最小值即可。

我个人认为这个问题的解肯定存在问题，我们不应该使用`Adam`优化器简单地搜索解，而是应该尽可能地搜索出最优解的位置。然而我并没有办法做到这个，目前最有效的方法仍然是简单的暴力搜索。

### C. Network Pruning

当神经元的数量增加的时候，网络的复杂性也在同时增加。为了进一步降低计算的复杂度，同时降低系统内存的负担，我们这里将采用剪枝算法对网络进行压缩。过程如下所示

首先建立神经网络并且训练网络直至网络表现出最优性能，然后计算`Average Percentage of Zeros, APoZ`。计算方式如下
$$
\begin{equation}
\begin{split}
APoZ_c^{(l)}=\frac{\sum_{s=1}^S \sum_{n=1}^N\delta(LeakyReLU(\boldsymbol o_{c,i}^{(l)}(n)))}{N\times S}
\end{split}
\tag{11}
\end{equation}
$$
使用验证数据集，即`Validation dataset`对每一个神经元做这样的计算。我们这里做的计算是用来计算零激活百分比的标度，这意味着在`LeakyReLU`的映射后神经元的输出在统计意义上为零。这里的作用其实是衡量神经元的重要程度，上述定义的第$$l$$层第$$c$$个神经元的`APoZ`。$$S$$是验证数据集的编号，$$N$$是第$$l$$层的神经元数量。接下来设置一个`APoZ`的阈值，大于这个阈值的就认为该神经元在摸鱼，大多数时候都是闲置的，不干活。这些要剪枝减掉，主要方法就是直接置零。对于小于阈值的神经元，就完整保留。根据经验建议设置阈值为$$0.8$$。最后**重新训练**网络以增强性能。

## IV. Numerical Results and Analysis

仿真环境是`Python 3.6.5`，软件包为`Tensorflow 1.1.0`和`Keras 2.2.2`，硬件环境为4个Intel i7-6700处理器和一块NVIDIA GTX1070显卡，8GB内存。显卡主要是用来辅助训练的，但是在实际使用的时候，决策过程运算都是CPU完成的。均采用Python进行运算，以搭建公平环境。数据生成过程如下

### A. Data Generation and Setup

我们将接收噪声协方差归一化为$$R=\sigma^2 I_Q$$，信道矩阵是独立同分布的高斯随机变量，方差为信道噪声与干扰。我们可以首先分离信道增益矩阵实部与虚部，然后拼成矢量作为深度网络的输入。训练样本和测试样本的大小分别为`50000`和`5000`。通过`cross-validation`来选择学习速率和`batch size`。

### B. Results and Analysis

这里就懒得写了自己看吧。反正中心思想就是说，`WMMSE`方法效果最好，监督与非监督学习都能达到类似的效果，但是时间上神经网络方法更短。





## V. Conclusion

`机翻警告`

在本文中，我们使用DNN模型来设计波束形成矩阵，与传统的WMMSE算法相比，这大大降低了计算复杂度，同时确保了性能。 基于加权和和速率的损失函数用于实现无监督学习，其实现了比监督学习方法更好的性能。 此外，我们测试了不同SNR和发射天线数量对DNN性能的影响。 结果表明，随着信噪比和发射天线数量的增加，DNN的性能下降，但仍然接近WMMSE。 最后，我们使用修剪方法在训练过程中重新加载预先训练的网络模型，并采用'APoZ'阈值方法来消除不活跃的神经元并压缩网络体积以最小化神经网络的计算复杂性。







