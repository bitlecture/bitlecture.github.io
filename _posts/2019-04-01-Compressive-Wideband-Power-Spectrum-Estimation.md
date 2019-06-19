---
layout: post
comments: true
title:  "Compressive Wideband Power Spectrum Estimation"
excerpt: "-"
date:   2019-04-01 12:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
---


# Compressive Wideband Power Spectrum Estimation

## Abstract

在一些应用中，例如认知无线电中的宽带频谱感知等之中，只有功率谱密度是我们感兴趣的内容，而没有必要恢复原信号。此外，高速`ADC`对于直接宽带频谱感知来说需要过高的功率（`too power hungry`）。这两个事实激励我们去调查压缩宽带功率谱感知，其包含了一个**压缩采样过程**，和一个可以从（亚奈奎斯特采样速率样本的）广义平稳信号恢复出功率谱密度的重建方法。这个方法与频谱盲采样不同，频谱盲采样的目的是重建原信号而不是功率谱密度，前面也说过在一些应用中并不需要我们恢复原信号，功率谱密度就是我们的目标。

本文中**首先**提出了一个基于周期性采样方法的解决方案和一个简单的最小二乘重建方法的解决方案。我们在时域和频域中对重建结果进行评估，**然后**我们测试了压缩采样中的两种可能的实现，分别是复高斯采样`complex Gaussian sampling`和多陪集采样`Multicoset Sampling`。当然我们主要是用了后面这个。基于所谓的最小稀疏标尺问题我们引入了一种新的多陪集采样方法。**接下来**我们分析了估计功率谱的统计特性。估计的均值和方差的计算允许我们计算功率谱估计值的分析归一化均方误差`NMSE`。

此外，当假设接收信号仅包含圆形复零均值独立同分布高斯噪声信号的时候，计算出的均值和协方差可以用于导出合适的检测阈值。仿真结果显示了我们提出的方法有一个相当好的表现（达到了期望值）。请注意我们的方法的带来的优势都是在没有对功率谱进行任何稀疏性约束的情况下产生的。

## I - Instruction

在近些年，宽带功率谱密度估计和感知成为了一个**信号处理**与**通信**领域中十分热门的话题。一个十分热门的应用就是认知无线电，其中未经允许的用户必须感知一个宽频谱范围，以便在建立通信链路的时候不会占据其他频段。一种可能的方法就是将整个宽带频谱划分为若干窄带信道，然后进行逐信道顺序感知。显然这种方法在频谱感知的过程中会引入显著的延迟。

*在一些文献[1]中，人们引入了一个滤波器组结构，以便在宽带状态下执行多通道频谱感知。类似的，文献[2],[3]中分别通过引入所谓的多频带联合检测和多频带感知时间自适应来优化多个窄带检测器组以改善认知无线电系统的聚合吞吐量。同样，由于需要大量的带通滤波器，这些方法效率都不咋地。*

另一种实现方法就是采用高速数模转换器直接扫描宽带频谱，

*例如文献[4]，其中用到小波来检测占用频带的边缘或边界。实际上前面也说了，高速`ADC`功率巨大[5]。为了减轻`ADC`的负担，许多研究者研究利用频谱的一些特定特征（例如频谱中的稀疏性或边缘频谱[6]-[8]）。这些特定的属性允许我们降低采样率（相比于奈奎斯特频率），同时还能在没有噪声的条件下实现完美的重建。在[9]中，多陪集采样被测试并提出来降低采样率，当我们所考虑的多频带信号在有限间隔的并集上具有频率支持的时候（这句没看懂照搬谷歌了）。鉴于先前对接收信号的频率已有了解，[9]中给出了精确的重建条件与明确地重建公式。不过很遗憾的是在许多应用中，认知无线电的频率并不是先前执导的，因此这个方法也是不适用的。为了解决这个问题，[7]和[10]提出了一个解决方案，基于没有任何先验知识，采用多陪集采样进行信号重建。相同思路的方法在[8]中也能找到，不过他们讨论的就是亚奈奎斯特采样在稀疏多频带的问题了。见于这几篇文章中讨论的问题是在不知道具体频率的条件下进行最小速率采样与重建，利用频谱的稀疏性。*

在这些工作中，我们已经发现大多数信号最小平均采样频率界，被称为`Landau lower bound`，相当于奈奎斯特频率与频率占用率的乘积。然而在最差的场景中，最小平均采样频率往往会是这个下界的两倍甚至奈奎斯特频率的两倍。请注意所有上述方法都可以转换为压缩采样的框架，其中信号重建可以采用稀疏恢复方法，例如最小绝对收缩和选择算子（`LASSO`）算法。还可以采用更为经典的方法，例如最小方差无失真响应（`MVDR`）方法或者多信号分类方法（`MUSIC`）。

不过上述方法都集中在了频谱估计上，目标都是完美地重建原始信号。事实上对于频谱感知的应用，只需要恢复功率谱（也就是说，功率谱密度），或者等效的自相关函数。

*基于亚奈奎斯特速率样本的功率谱估计算法已经在[13]和[14]中通过自相关函数方法而得到了发展。在文献[15]中，一种基于多陪集采样功率谱估计方法被提出，这利用了一个宽平稳信号对应于频域表示的对角协方差矩阵的事实（这他妈又是啥）。这个发现被用于[15]来建立一个超定方程组，将压缩测量的频域统计与信号的频域统计相关联。*

另一种被称为是互质采样的方法在[16]被提出，该方法目标是通过两个均匀的亚奈奎斯特采样器来估计埋在噪声中的正弦信号的频率，采样周期是奈奎斯特周期的互质倍数。

本文重点研究了有效的功率谱重建技术，并以为此设计有效的周期性亚奈奎斯特采样过程。在[17]中也被称为是功率谱盲采样`PSBS`。理论上讲这种方法能够利用最小二乘法通过周期采样装置不同的输出之间的互相关来完美地重建宽平稳信号的未知功率谱。最小二乘法需要一些秩的条件，这将是指导采样装置实现的重要条件。本文中，可以采用基于随机调制波形的采样技术，但是主要关注的仍然是多陪集方法。我们基于最小稀疏标尺问题设计了一种新的多陪集采样的实现，还调查研究了估计的功率谱的理论统计特性，用于分析均值与协方差，这对于分析指定`NMSE`是有效的。此外，还可以通过假设接受信号仅是圆形复数零均值高斯噪声来导出用于评估特定频率的信号的存在与否的检测阈值。所有提出的方案都是通过分析与仿真来比较的。通常，我们开发的采样过程可以通过利用频谱互相关性显著地降低采样速率需求，而不对功率谱做任何的稀疏性约束。

## II - System Model and Problem Statement

`2019-4-2 10:18:02`今天写完这部分就不写了！

定义$$x(t)$$是宽平稳模拟信号，并认为是一个复数值。这个信号是带限的，带宽是$$\frac{1}{T}$$（这个就是奈奎斯特频率，大小等于采样频率的一半）。我们考虑这样一个频谱感知的应用场景，任务是感知$$x(t)​$$的功率谱密度。图1描绘了部署的采样设备，可以被认为是压缩采样中一种模拟信息转换器（`Analog to Information Converter, AIC`）。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/XDPFig1.png" style="display: inline-block;" width="500"/>
</div>


但是请注意这个采样设备能够对任何`AIC`实现建模，例如[18]和[19]中提出的那些。采样装置此处有$$M$$个分支，其中分支使用复值周期性波形$$p_i(t)$$来调制$$x(t)$$，后面接周期性积分转储装置（因此速率等于奈奎斯特速率的$$\frac{1}{N}$$倍）。请注意观察输出是数字形式，因为每隔$$NT$$积分输出就相当于每隔$$NT​$$给了一个采样点。这个采样方式可能会保存更多的信息。

第$$i$$个分支的第$$k$$个采样索引可以被描述为
$$
\begin{equation}
\begin{split}
y_i[k]=&\frac{1}{NT}\int_{kNT}^{(k+1)NT}p_i(t)x(t)dt\\=&\frac{1}{T}\int_{kNT}^{(k+1)NT}c_i(t-kNT)x(t)dt
\end{split}
\tag{1}
\end{equation}
$$

这里的$$c_i(t)$$是$$\frac{1}{N}p_i(t)$$的单个周期，比如

$$c_i(t)=\frac{1}{N}p_i(t)\:for\:0\leq t \leq NT\:and \:c_i(t)=0\: elsewhere​$$

假设$$c_i(t)$$是一个分段常数函数，每隔长度$$T$$取值改变，上式可以改写为
$$
\begin{equation}
\begin{split}
y_i[k]=&\sum_{n=0}^{N-1}c_i[-n]\frac{1}{T}\int_{kNT}^{(k+1)NT}x(t)dt\\=&\sum_{n=0}^{N-1}c_i[-n]x[kN+n]\\=&\sum_{n=1-N}^{0}c_i[-n]x[kN-n]
\end{split}
\tag{2}
\end{equation}
$$
其中$$x[n]$$可以被看作是周期$$T$$内的积分转储过程的输出，由于其高度复杂性，此处并未明确计算。此周期性采样的平均采样率是奈奎斯特频率与$$\frac{M}{N}$$的乘积，我们一般都会取$$M<N$$来保证复杂性更低。这个采样设备实际上与[8]中提到的调制宽带转换器相似，其中$$c_i[n]​$$的取值是随机生成的，例如采用复数高斯采样或者随机二进制采样。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/XDPFig2.png" style="display: inline-block;" width="500"/>
</div>
这个图示结构是第一个图的数字实现方式之一，含有高速积分转储过程，后接$$M$$分支，每个分支都有一个数字滤波操作，然后是降采样部分。这个图抢掉了一个事实，即第二个流程相当于数字滤波过程，滤波器就是$$c_i[n]$$，后接$$N$$倍抽取。滤波器原理我就懒得写了。就是卷积。

本文的目标就是重建$$x(t)$$的功率谱密度，基于我们上述采样得到的$$\{y_i[k]\}$$。由于$$x[n]$$是$$x(t)$$经过奈奎斯特频率积分转储设备得到的，那么$$x[n]$$的功率谱密度仅仅是原功率谱密度的周期性延拓而没有产生混叠。因此，我们说两者之间是互相决定的。那么接下来我们主要讨论的就是如何重建$$x[n]$$的功率谱密度。功率谱密度定义式

$$P_x(\omega)=\sum\limits_{n=-\infty}^{\infty}r_x[n]e^{-jn\omega}$$



这个定义式我不再进行详细介绍了。这个的主要贡献在于我们利用了$$y​$$的所有$$M^2​$$个不同的互谱，折将实现速率压缩而不会在$$x(t)​$$引入任何的稀疏性约束。互谱密度在下面定义

$$P_{y_i,y_j}(\omega)=\sum\limits_{k=-\infty}^{\infty}r_{y_i,y_j}[k]e^{-jk\omega}$$

这些$$r_{y_i,y_j}$$集合量可以通过它们的样本平均值来计算，这还可以让我们计算出$$P_{y_i,y_j}​$$。下面我们将介绍一种时域方法，在给出输出互相关序列的情况下重建出输入自相关序列。然后我们使用频域上的估计方法就可以估计出功率谱密度，同样在给定采样输出互谱密度的条件下。

## III - Time-Domain Reconstruction Approach

时域重建方法概述

### A. Reconstruction Analysis

这一部分我们提出一种方法来根据采样输出互相关序列重建输入自相关序列。由于我们已知$$y_i[k]=z_i[kN]$$，则经过推导容易知道
$$
\begin{equation}
\begin{split}
r_{y_i,y_j}[k]=r_{z_i,z_j}[kN]
\end{split}
\tag{3}
\end{equation}
$$
显然又有
$$
\begin{equation}
\begin{split}
r_{z_i,z_j}[n]=r_{c_i,c_j}[n]\star r_x[n]=\sum_{m=-N+1}^{N-1}r_{c_i,c_j}[m]r_x[n-m]
\end{split}
\tag{4}
\end{equation}
$$
实际上在已知$$c_i[n]$$序列的条件下，这里的$$r_{c_i,c_j}[n]$$序列也是很容易得到的
$$
\begin{equation}
\begin{split}
r_{z_i,z_j}[n]=c_i[n]\star c^*_j[-n]=\sum_{m=-N+1}^{0}c_i[m]\star c^*_j[m-n]
\end{split}
\tag{5}
\end{equation}
$$
这样就可以建立$$r_{y_i,y_j}$$与$$r_x​$$之间的关系
$$
\begin{equation}
\begin{split}
r_{y_i,y_j}[k]=\sum_{l=0}^1\boldsymbol{r}_{c_i,c_j}^T[l]\boldsymbol{r}_x[k-l]
\end{split}
\tag{6}
\end{equation}
$$
其中定义
$$
\begin{equation}
\begin{split}
\boldsymbol{r}_{c_i,c_j}[0]=&\left[ r_{c_i,c_j}[0],r_{c_i,c_j}[-1],...,r_{c_i,c_j}[-N+1] \right]^T\\
\boldsymbol{r}_{c_i,c_j}[1]=&\left[ r_{c_i,c_j}[N],r_{c_i,c_j}[N-1],...,r_{c_i,c_j}[1] \right]^T\\
\boldsymbol{r}_{x}[k]=&\left[ r_{x}[kN],r_{x}[kN+1],...,r_{x}[(k+1)N-1] \right]^T
\end{split}
\tag{7}
\end{equation}
$$
通过级联$$M^2​$$个不同的互相关函数$$r_{y_i,y_j}[k]​$$，我们获得了$$M^2\times 1​$$的向量$$\boldsymbol{r}_y[k]​$$。这样就可以继续改写$$(6)​$$中的结果。这里要不就先不写了。

考虑到$$x[n]$$的带限性，$$\boldsymbol{r}_y[k]$$基本上没有限制。然而在许多实际的场景中，只有在$$-L\leq k\leq L$$的范围内的时候才有较为显著的取值，其他地方的取值都是接近零的。这里的$$L$$是一个可自定义调整的参数，我们想取多大就取多大。于是我们稍微放松一点对带限的条件，并假设$$\boldsymbol{r}_y[k]$$是严格限制在$$-L\leq k\leq L$$范围内的。此时我们认为$$\boldsymbol{r}_x[k]$$有着与$$\boldsymbol{r}_y[k]$$相同的限制条件，$$-L\leq k\leq L$$。

不如咱们就把这些都收纳到一个矩阵里面
$$
\begin{equation}
\begin{split}
\boldsymbol{r}_{y}=&\left[ \boldsymbol{r}^T_{y}[0],\boldsymbol{r}^T_{y}[1],...,\boldsymbol{r}^T_{y}[L],\boldsymbol{r}^T_{y}[-L],...,\boldsymbol{r}^T_{y}[-1] \right]^T\\
\boldsymbol{r}_{x}=&\left[ \boldsymbol{r}^T_{x}[0],\boldsymbol{r}^T_{x}[1],...,\boldsymbol{r}^T_{x}[L],\boldsymbol{r}^T_{x}[-L],...,\boldsymbol{r}^T_{x}[-1] \right]^T
\end{split}
\tag{8}
\end{equation}
$$
标记一下，输出自相关矩阵的大小为$$(2L+1)M^2​$$，输入阵大小为$$(2L+1)N​$$。我们再深入一点介绍另一个观察值，其中一个是基于上述$$(7)​$$的定义，我们根据$$0​$$值的分布，可以得出
$$
\begin{equation}
\begin{split}
\boldsymbol{r}_{y}=\boldsymbol{R}_{c}\boldsymbol{r}_{x}
\end{split}
\tag{9}
\end{equation}
$$
其中这个$$\boldsymbol{R}_{c}$$是一个$$(2L+1)M^2\times (2L+1)N$$矩阵，定义为
$$
\begin{equation}
\begin{split}
\boldsymbol{R}_{c}=\left[\begin{matrix} \boldsymbol{R}_{c}[0]& & & &\boldsymbol{R}_{c}[1] \\\boldsymbol{R}_{c}[1]&\boldsymbol{R}_{c}[0]\\ & \boldsymbol{R}_{c}[1]&\boldsymbol{R}_{c}[0]\\ & &...&...\\ & & & \boldsymbol{R}_{c}[1]&\boldsymbol{R}_{c}[0] \end{matrix}\right]
\end{split}
\tag{10}
\end{equation}
$$


如果$$\boldsymbol{R}_{c}$$是一个满秩的矩阵，这个矩阵就是可以解的。显然这个时候需要$$M^2\geq N​$$。

上述定义$$(9)$$的逆问题可以被进一步简化，观察$$\boldsymbol{R}_{c}$$可以知道这是一个循环矩阵，块大小为$$M^2\times N$$，很容易被转化为一个块对角矩阵。这个可以用$$2L+1$$点离散傅里叶变换完成。
$$
\begin{equation}
\begin{split}
\boldsymbol{R}_{c}=(\boldsymbol{F}_{2L+1}^{-1}\otimes \boldsymbol{I}_{M^2})\boldsymbol{Q}_{c}(\boldsymbol{F}_{2L+1}\otimes \boldsymbol{I}_{N})
\end{split}
\tag{11}
\end{equation}
$$
这里的$$\boldsymbol{F}_{2L+1}​$$是$$(2L+1)\times (2L+1)​$$的离散傅里叶变换矩阵。$$\boldsymbol{Q}_{c}=diag\{\boldsymbol{Q}_{c}(0),\boldsymbol{Q}_{c}(2\pi\frac{1}{2L+1}),...,\boldsymbol{Q}_{c}(2\pi\frac{2L}{2L+1})\}​$$，其中
$$
\begin{equation}
\begin{split}
\boldsymbol{Q}_{c}(\omega)=\sum_{k=0}^1\boldsymbol{R}_{c}[k]e^{-jk\omega}
\end{split}
\tag{12}
\end{equation}
$$
如果我们定义一个$$\boldsymbol{q}_{x}=(\boldsymbol{F}_{2L+1}^{-1}\otimes \boldsymbol{I}_{N})\boldsymbol{r}_{x}$$和$$\boldsymbol{q}_{y}=(\boldsymbol{F}_{2L+1}^{-1}\otimes \boldsymbol{I}_{M^2})\boldsymbol{r}_{y}$$，那么就可以得到
$$
\begin{equation}
\begin{split}
\boldsymbol{q}_{y}=\boldsymbol{Q}_{c}\boldsymbol{q}_{x}
\end{split}
\tag{13}
\end{equation}
$$
相当于完成了一个归一化。

这里我们的$$\boldsymbol{q}_{y}$$是一个$$\boldsymbol{r}_{y}[k]$$的样本平均，而$$\boldsymbol{r}_{x}[k]$$和$$\boldsymbol{q}_{x}$$就是一个待估计量。注意这里我们通过一些数学手段将时域上的任务无意间转化到了频域上。不过考虑到方法的问题，实际上仍然是时域上的处理。实际上我们也可以认为这个是频域处理，至于为什么我们称其为时域处理，是因为要与后面的频域处理作出区分。当我们能够估计$$\boldsymbol{q}_{x}$$之后，我们就可以根据上面的结果来估计$$\boldsymbol{r}_{x}$$，进而得到功率谱密度
$$
\begin{equation}
\begin{split}
\boldsymbol{s}_{x}=\boldsymbol{F}_{(2L+1)N}\boldsymbol{r}_{x}
\end{split}
\tag{14}
\end{equation}
$$
这里$$\boldsymbol{s}_{x}=\left[ P_x(0),P_x(2\pi\frac{1}{(2L+1)N}),...,P_x(2\pi\frac{(2L+1)N-1}{(2L+1)N}) \right]^T$$

### B. Alternative Time-domain Approach

在这一小节里面，我们提出了另一个不同的时域重建方法。我们从重写矩阵表达$$(2)$$开始
$$
\begin{equation}
\begin{split}
\boldsymbol{y}[k]=\boldsymbol{C}\boldsymbol{x}[k]
\end{split}
\tag{15}
\end{equation}
$$
我们把图再看一下

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/XDPFig2.png" style="display: inline-block;" width="500"/>
</div>

这里很显然把$$x,y$$都用向量表示了。我们这里还给​$$\boldsymbol{C}$$起了一个牛逼的名字，叫压缩采样矩阵。从这个图捋一下过程，输入的是一个模拟信号$$x(t)$$，经过积分转储之后得到的是一段离散序列，离散序列经过不同的成型滤波器之后，得到序列做$$N$$倍下采样就得到了$$M$$个输出序列$$y$$。接下来计算互相关序列，得到
$$
\begin{equation}
\begin{split}
\boldsymbol{R}_{y}[0]=\boldsymbol{C}\boldsymbol{R}_x^a\boldsymbol{C}^H
\end{split}
\tag{16}
\end{equation}
$$
由于我们分析的是宽平稳信号，因此相关阵有`Toeplitz`结构。不过输出就已经不能说是宽平稳信号了，因此就未必有这种特性了——压缩采样的特性。因此理论上利用$$\boldsymbol{R}_{y}[0]$$的每一列来估计$$\boldsymbol{R}_x^a$$的其中一列就是可能的了。

基于上式，我们可以得到
$$
\begin{equation}
\begin{split}
\boldsymbol{r}_{y}[0]=vec(\boldsymbol{R}_{y}[0])=(\boldsymbol{C}^*\otimes \boldsymbol{C})vec(\boldsymbol{R}_{x}^a)
\end{split}
\tag{17}
\end{equation}
$$
其中这个$$\otimes$$代表克罗内克积。这个运算相当复杂。后面矩阵运算实在是看不懂了，有机会再说吧。

## IV - Frequency-Domain Reconstruction Approach

这里直接讲频域重建方法了。提出频域重建方法主要是因为频谱盲采样`SBS`。`SBS`也是从频域出发，不过主要是面向频谱重建而不是功率谱重建。










## Reference

[1] 

