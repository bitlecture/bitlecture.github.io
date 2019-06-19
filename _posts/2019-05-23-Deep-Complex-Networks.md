---
layout: post
comments: true
title:  "Deep Complex Networks"
excerpt: "-"
date:   2019-05-23 12:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
---

[TOC]



## Abstract

目前绝大多数的深度学习的技术或模型结构都是基于实数计算与表达的，但是近期在`RNN`上的研究（以及一些较老的基础理论分析）表明，复数可能会有更为丰富的表现力，并且有可能产生富有噪声鲁棒性的应用。尽管如此，由于缺少基本的`building blocks`，复值深度神经网络已经被边缘化了，这有可能是缺乏相关研究人员，或者没有成熟的理论研究的需求。

这篇文章给出了复值深度网络的最基础组成成分，并且尝试将上述结构应用在卷积前馈网络结构与卷积`LSTM`网络结构中。其中的所有操作都是基于复数的。我们证明，复值网络与实值网络同样具有竞争力，并在计算机视觉与音乐等应用中进行了测试，达到了目前研究相同的水平。

## 1. Introduction

近期研究注重于解决深度学习中仍存在的问题，例如标准化`Normalization`技术、门控网络、残差网络等。

基于复数表示的作用已经开始受到越来越多的关注，因为它们有可能实现更容易的优化，可能有更为优秀的泛化特征，甚至有更快的学习速度，且对于噪声有更强的鲁棒性。有研究表明`RNN`中采用复数可以让模型有更为优秀的表现力。

为了利用复数表示提供的优势，我们提出了复值深度神经网络的构建组建的一般公式，并且应用在上述的的两种模型作为实例。本文的实质性贡献为

1. 复数的批量标准化公式`Batch Normalization Formulation`
2. 复权重初始化
3. 不同的复数的基于`ReLU`的激活函数的比较
4. 多乐器音乐转录数据集的最新成果
5. 基于`TIMIT`数据集的语音频谱预测的最新成果

我们对深度复值网络的健全性做了检查，确保在标准图像分类的有效性，尤其是基于`CIFAR-10/100`的数据集的结果。我们在其他上述的数据集中的验证结果表明学习复值表示会产生与实值相似的性能。



## 2. Motivation and Related Work

*这一段懒得看了，机翻了。*

`机翻警告`

从计算，生物和信号处理的角度来看，使用复杂参数具有许多优点。 从计算的角度来看，Danihelka等人。 （2016）已经表明，使用复数的全息缩减表示（Plate，2003）在从关联记忆中检索信息的背景下在数值上是有效和稳定的。 Danihelka等。 （2016）通过添加到存储器轨迹中，在关联存储器中插入键值对。 虽然通常不被视为这样，但残余网络（He et al。，2015a; 2016）和Highway Networks（Srivastava et al。，2015）具有与关联存储器类似的架构：每个ResNet残余路径计算随后插入的残差 -  通过汇总到身份连接提供的“记忆”。鉴于剩余网络在几个基准测试中取得了巨大成功，并且它们与联想记忆的功能相似，将两者结合在一起似乎很有意思。 这促使我们在残余网络中加入复杂的权重和激活。 它们一起提供了一种机制，通过该机制可以在每个残余块中检索，处理和插入有用信息。

正交权重矩阵对RNN中众所周知的消失和扩散梯度问题提供了新的攻角。 单一RNN（Arjovsky等，2015）基于单位权重矩阵，它是正交权重矩阵的复杂推广。 与它们的正交对应物相比，酉矩阵提供更丰富的表示，例如能够实现离散傅里叶变换，并因此能够发现光谱表示。 Arjovsky等。 （2015）显示了这种类型的递归神经网络对玩具任务的潜力。 Wisdom等。 （2016）为学习单一矩阵提供了更一般的框架，并将他们的方法应用于玩具任务和现实世界的语音任务。

在神经网络中使用复杂权重也具有生物学动机。 Reichert和Serre（2013）提出了一种生物学上可信的深层网络，允许人们使用复值神经元单元构建更丰富，更通用的表示。复数值公式允许一个表达神经元输出的发射率和其活动的相对时间。复杂神经元的振幅代表前者，其相位代表后者。具有相似相位的输入神经元被称为同步，因为它们建设性地添加，而异步神经元破坏性地增加并因此相互干扰。这与深度前馈神经网络中使用的门控机制有关（Srivastava等，2015; van den Oord等，2016a; b）和递归神经网络（Hochreiter和Schmidhuber，1997; Cho et al。，2014; Zilly et al。，2016），因为这种机制学会了同步网络在给定的前馈层或时间步长传播的输入。在基于深度选通的网络的情况下，同步意味着输入的传播，其控制门同时保持高值。这些控制门通常是S形函数的激活。这种考虑相位信息的能力可以解释在复现神经网络的背景下结合复值表示的有效性。

从生物学的观点来看，相位分量不仅是重要的，而且从信号处理的角度来看也是如此。 已经表明，语音信号中的相位信息影响它们的可懂度（Shi等，2006）。 Oppenheim和Lim（1981）也表明，图像阶段存在的信息量足以恢复大小编码的大部分信息。 实际上，阶段在对象对形状，边缘和方向进行编码时提供对对象的详细描述。

最近，Rippel等人。 （2015）利用卷积神经网络的傅立叶频谱表示，提供了一种参数化频谱域中的卷积核权重，并对信号的频谱表示进行汇集的技术。 然而，作者避免执行复值卷积，而是从空间域中的实值内核构建。 为了确保谱域中的复杂参数化映射到实值核上，作者对谱域权重施加了共轭对称约束，这样当对它们应用逆傅立叶变换时，它只产生实数的内核。

正如Reichert和Serre（2013）所指出的，已经研究了复值神经网络的使用（Georgiou和Koutsougeras，1992; Zemel等，1995; Kim和Adalı，2003; Hirose，2003; Nitta，2004）。早在最早的深度学习突破之前（Hinton等，2006; Bengio等，2007; Poultney等，2007）。最近Reichert和Serre（2013年）;布鲁纳等人。 （2015）; Arjovsky等。 （2015）; Danihelka等。 （2016）; Wisdom等。 （2016）通过为使用复值深度网络提供理论和数学动机，试图更多地关注深度复杂神经网络的有用性。然而，据我们所知，除了一些尝试之外，大多数使用复杂价值网络的近期作品已经应用于玩具任务。事实上，（Oyallon和Mallat，2015; Tygert等，2015; Worrall等，2016）在视觉任务中使用了复杂的表示。 Wisdom等。 （2016）还进行了现实世界的语音任务，包括预测未来短时傅立叶变换帧的对数幅度。在自然语言处理中，（Trouillon等，2016; Trouillon和Nickel，2017）使用了复值嵌入。要开发适当的工具和训练具有复值参数的深度神经网络的一般框架，仍有许多工作要做。

鉴于使用复值表示的令人信服的理由，缺少这样的框架工作代表了机器学习工具的一个空白，我们通过为深度复值神经网络提供一组构建块来填补它们，使它们能够获得有竞争力的结果 与他们真实价值的同行在现实世界的任务。

## 3. Complex Building Blocks

`手工翻译`

 本部分将介绍工作的核心，介绍实现复数值深度神经网络的`building blocks`的数学框架。

### 3.1 REPRESENTATION OF COMPLEX NUMBERS

复数的表示方法是
$$
\begin{equation}
\begin{split}
z=a+bi
\end{split}
\tag{1}
\end{equation}
$$

这个和我们课程介绍的一样，不过多介绍。考虑一个典型的实值`2-D`卷积层，其中有$$N$$个特征特征映射，我们采用前$$N/2$$个特征映射来表示实部，剩下的$$N/2$$来映射虚部。因此对于$$N_{in}$$个输入，$$N_{out}$$个输出特征的$$m\times m$$卷积核，我们的权重矩阵将有$$N_{in}\times N_{out}\times m\times m$$个复权重。

> 说句题外话，我今天才知道卷积操作是如何等效为矩阵的。图像的卷积是卷积核在图像上的平移乘积过程，但是运算的时候并非是这样运算的，为了加速可以将卷积核内的内容直接展开成为一个向量，然后与卷积核展开的向量做向量积，得到一个标量。如果把所有需要卷积的内容排成一个大矩阵，这个运算就成为了矩阵乘法，这样就可以加速运算。卷积后的向量，我们就称之为`feature map`。一般的，一个卷积核对应一个`feature map`。

### 3.2 COMPLEX CONVOLUTION

为了推广实卷积到复卷积，这里给出权重矩阵的新定义
$$
\begin{equation}
\begin{split}
\boldsymbol W=\boldsymbol A+\boldsymbol Bi
\end{split}
\tag{2}
\end{equation}
$$
也就是二维滤波器的新定义方法。与此相同的是向量的定义
$$
\begin{equation}
\begin{split}
\boldsymbol h=\boldsymbol x+\boldsymbol yi
\end{split}
\tag{3}
\end{equation}
$$
这里标黑的都是实数向量或矩阵。以上述二者的卷积为例

复卷积的定义为
$$
\begin{equation}
\begin{split}
\boldsymbol W *\boldsymbol h=(\boldsymbol A*\boldsymbol x-\boldsymbol B*\boldsymbol y)+i(\boldsymbol B*\boldsymbol x+\boldsymbol A*\boldsymbol y)
\end{split}
\tag{4}
\end{equation}
$$

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/ComplexNetworks1.png" style="display: inline-block;" width="650"/>
</div>


如图所示我们将运算写成矩阵的形式就会得到
$$
\begin{equation}
\begin{split}
\left[\begin{matrix} \mathfrak R (\boldsymbol W*\boldsymbol h)\\ \mathfrak I(\boldsymbol W*\boldsymbol h) \end{matrix}\right]=\left[\begin{matrix} \boldsymbol A &-\boldsymbol B\\ \boldsymbol B &\boldsymbol A \end{matrix}\right]*\left[\begin{matrix} \boldsymbol x\\ \boldsymbol y \end{matrix}\right]
\end{split}
\tag{5}
\end{equation}
$$

### 3.3 COMPLEX DIFFERENTIABILITY

这里需要考虑到复链式法则。我贴个图

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/ComplexNetworks6.3.png" style="display: inline-block;" width="600"/>
</div>


为了可以实现复数网络的反向传播算法，就需要我们可以找到一个分别在实数和虚数部分都可微的代价函数和激活函数。

通过限制激活函数为复可微或全纯函数，我们直接限制了激活函数的使用。另外关于“全纯”的问题，我在这里再贴一个图。全纯函数的实部和虚部都是调和的，满足柯西黎曼方程。



### 3.4 COMPLEX-VALUED ACTIVATIONS

#### 3.4.1 MODRELU

很多激活函数都被用于复数的尝试中，如下是其中的一种，被称为`modReLU`
$$
\begin{equation}
\begin{split}
modReLU(z)=ReLU(\mid z \mid+b)e^{i\theta_z}=\max \left( (\mid z \mid+b)\frac{z}{\mid z\mid},0 \right)
\end{split}
\tag{6}
\end{equation}
$$
其中$$\theta_z$$是$$z$$的相位，$$b$$是一个可学习参数，是一个实数。如果括号内是正的，则输出该复数方向的一定倍数，否则输出$$0$$。这里的$$b$$参数的主要目的是创造一个死区，半径为$$b$$。该激活方法设计的初衷是可以保留原来的方向（或者说相位），否则复数的结构可能会被严重破坏。由于这个不满足柯西黎曼方程，因此这个函数不是全纯的。这里有机会贴一张相关的测试图。



#### 3.4.2  CRELU AND zRELU

$$\mathbb C ReLU$$是`Complex ReLU`，是对实部和虚部分别应用$$ReLU$$的结果。
$$
\begin{equation}
\begin{split}
\mathbb CReLU(z)=ReLU(\mathbb R(z))+iReLU(\mathbb I(z))
\end{split}
\tag{7}
\end{equation}
$$
当虚部和实部同时为严格正或负时满足柯西黎曼方程，同样的测试结果有空会贴上

另一个相似的激活函数是$$zReLU$$函数，
$$
\begin{equation}
\begin{split}
zReLU(z)=z,\:where\:\theta_z\in[0,\pi/2],else \:\:0
\end{split}
\tag{8}
\end{equation}
$$
上述结果的测试如下

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/ComplexNetworksTab1.png" style="display: inline-block;" width="600"/>
</div>

以及

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/ComplexNetworksTab2.png" style="display: inline-block;" width="600"/>
</div>

### 3.5 COMPLEX BATCH NORMALIZATION

深度网络中应用`Batch Normalization`来加速学习速度，在一些情况下，这种标准化对模型的优化起很大作用。对于复数的正则化方法，我们给出公式

首先说明，对复数做高斯归一化，实数方法是不够的，因为不能保证实部虚部有相同的方差，因而造成椭圆形的分布，有较高的偏心率。

我们最终将模型设计为白化`2D`向量的模型，通过两个部分分别的标准差缩放数据，这样的操作可以通过协方差矩阵的逆矩阵与去心数据相乘完成
$$
\begin{equation}
\begin{split}
\tilde {\boldsymbol x} = (\boldsymbol V)^{-\frac 1 2}(x-\mathbb E[\boldsymbol x])
\end{split}
\tag{9}
\end{equation}
$$
其中协方差矩阵可以表示为
$$
\begin{equation}
\begin{split}
\boldsymbol V =\left( \begin{matrix} Cov(\mathfrak R,\mathfrak R)& Cov(\mathfrak R,\mathfrak I)\\Cov(\mathfrak I,\mathfrak R) &Cov(\mathfrak I,\mathfrak I)\end{matrix} \right)
\end{split}
\tag{10}
\end{equation}
$$
这个矩阵的逆矩阵很容易求，而且本身有着较好的（半）正定性。最终结果有着符合要求的均值与（协）方差。

归一化过程允许人们对虚部和实部进行去相关操作，防止两部分之间出现互适应现象从而降低过拟合的风险。我们的批量归一化操作方法描述为
$$
\begin{equation}
\begin{split}
BN(\tilde {\boldsymbol x}) =\gamma \tilde {\boldsymbol x}+\beta 
\end{split}
\tag{11}
\end{equation}
$$
其中
$$
\begin{equation}
\begin{split}
\gamma = \left( \begin{matrix} \gamma_{rr}&\gamma{ri}\\ \gamma_{ri}&\gamma_{ii} \end{matrix} \right)
\end{split}
\tag{12}
\end{equation}
$$
初始化时均初始化为$$1/\sqrt{2}$$，$$\beta=0+0i$$。优化过程采用带动量的滑动平均。

### 3.6 COMPLEX WEIGHT INITIALIZATION

当我们没有部署批量归一化的时候，正确的初始值是一个非常重要的降低梯度爆炸或消失的因素。一个复数权重可以表示为
$$
\begin{equation}
\begin{split}
W=\mid W\mid e^{i\theta}=\mathfrak R\{ W \}+i\mathfrak I\{ W \}
\end{split}
\tag{13}
\end{equation}
$$
计算方差为
$$
\begin{equation}
\begin{split}
var(W)=\mathbb E[WW^*]-\mathbb E^2[W]
\end{split}
\tag{14}
\end{equation}
$$
当均值为$$0$$的时候，这个结果就会自然地**收敛到前面的式子**。我们并不知道这个方差是多少，但是我们知道$$var(|W|)$$，
$$
\begin{equation}
\begin{split}
var(|W|)=\mathbb E[|W||W|^*]-\mathbb E^2[|W|]
\end{split}
\tag{15}
\end{equation}
$$
可以将两式合并得到
$$
\begin{equation}
\begin{split}
var(|W|)=var(W)-\mathbb E^2[|W|]
\end{split}
\tag{16}
\end{equation}
$$


因此可以通过瑞利分布得到原结果
$$
\begin{equation}
\begin{split}
E[|W|]&=\sigma\sqrt \frac \pi 2\\
Var(|W|)&=\frac{4-\pi} 2 \sigma^2
\end{split}
\tag{17}
\end{equation}
$$
即
$$
\begin{equation}
\begin{split}
var(W)=2\sigma^2
\end{split}
\tag{18}
\end{equation}
$$
这是很简单的推导过程。

根据前人不同的研究成果，这里的$$\sigma$$其实可以较为自由地选取，在不同条件下得到的结果也是不同的。 方差显然只由$$\sigma $$这一个参数决定，而且与相位没有关系，所以我们可以使用$$-\pi\sim \pi$$均匀分布来初始化相位。

学习去相关的特征有利于泛化以及快速学习。

> 实话讲初始化这一部分我就没看懂

### 3.7  COMPLEX CONVOLUTIONAL RESIDUAL NETWORK

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/ComplexNetworks6.1.png" style="display: inline-block;" width="500"/>
</div>

















