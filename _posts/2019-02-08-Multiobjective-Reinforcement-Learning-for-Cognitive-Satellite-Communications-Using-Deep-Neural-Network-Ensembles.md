---
layout: post
comments: true
title:  "Multiobjective Reinforcement Learning for Cognitive Satellite Communications Using Deep Neural Network Ensembles"
excerpt: "-"
date:   2019-02-08 14:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
---

## 备注

- 这个文章看起来英文水平比较好
- `2019-2-8 20:40:19`放你妈的狗屎这什么垃圾英语
- 用了巨多从句，整的谷歌都翻译不出来 = =、
- 这篇论文话好多
- 我发现每天的工作就是打开`Acrobat`，打开`Typora`，打开谷歌翻译和`Github`，然后开始复制粘贴和瞎BB
- 可以说本文作者很喜欢故弄玄虚了。
- 神经网络被他们当做了无敌的非线性拟合器，ε=(´ο｀*)))
- 哎我真是佛了，三层的全连接神经网络被他说的跟啥似的，这人是没见过神经网络还是故意这样说的
- 自卖自夸的本领我也是服气的

## 摘要 - Abstract

未来的航天通信子系统将受益于人工智能相关算法控制的软件无线电。本文中我们提出了一种新的无线电资源分配算法，利用多目标强化学习与人工神经网络结合，管理可用资源和冲突任务为基础的目标。数千种可能的无线电参数组合的性能不确定性，以及无线电信道随时间的动态行为，产生连续的多维状态-动作空间，需要固定大小的存储的连续状态-动作映射，而不是传统的离散映射。此外，操作决策应与状态分离，以允许在线学习、性能监控和资源分配预测。所提出的方法利用了作者之前关于通过“虚拟环境探索”约束预测性能较差的决策研究。

仿真结果显示了不同通信任务中该方法的性能，为未来的研究参考提供了数值准确的基准。该方法也构成了核心认知引擎概念验证的一部分，交付给了NASA的John H. Glenn的研究中心在国家空间站上的SCaN Testbed无线电系统。

## I - 引入 - INTRODUCTION

> 你妈的废话又臭又长

2012年，由NASA John H. Glenn研究中心的**空间通信与导航**(`Space Communications and Navigation, SCaN`)小组领导的一个研究项目为**国际空间站**(`International Space Station, ISS`)提供了一个通信研究平台。该平台被称为`SCaN Testbed`，是由三个软件定义的无线电系统组成，旨在促进未来航空航天应用的在轨通信研究。天基通信系统的下一个前沿(`frontier`)是开发和测试**认知引擎**(`cognitive engines, CEs`)，利用在轨软件定义无线电(`Software-Deﬁned Radios, SDRs`)的潜力，用于未来的太空探索任务。

预计下一代天基通信系统将提供更高的灵活性，以便在具有挑战性的环境中更有效地运行，包括轨道动力学和大气和/或空间天气，或者当要求航天器在不可预测的条件下运行时。因此，需要**认知引擎**有效地分配资源以实现若干目标，每个目标具有一定的优先级，同时通信信道动态地改变。 **认知引擎**应考虑通信系统对其他航天器子系统的资源消耗的影响，同时分配多个不同的资源以实现多个目标。

认知无线电(`Cognitive Radio, CR`)拥有**认知引擎**，它可以利用跨不同网络层的环境感知，并且能够自主地执行**感知**、**学习**和**推理**(`perception, learning, and reasoning`)活动，以便根据当前节点的**硬件和软件能力**、**信道条件**和**操作需求**优化**资源分配**。

目前已经有一些简单地自适应技术部署到现实的应用中了，这些被作为未来完全认知系统的基石。例如，作为DVB-S2标准的一部分，**自适应编码调制**(`adaptive coding and modulation, ACM`)方案在卫星电视接收机的信号衰落事件期间调整无线电参数。另一个例子是**动态信道接入**的**频谱感应**(`Spectrum Sensing, SS`)，其中临时未使用的频谱被重新用于不同的应用。

过去已经使用认知无线电为案例研究提出了几种机器学习的技术。机器学习支持在线学习，这是认知引擎的核心功能。一些人已经具体研究过学习问题，例如基于机器学习的认知无线电问题与资源分配等。

这些自适应技术单独运行的时候效果很好，例如**自适应编码调制**有助于缓解衰落问题（这在[上一篇文章](https://psycholsc.github.io/notes/2019/01/22/Precoding-Scheduling-and-Link-Adaptation-in-Mobile-Interactive-Multibeam-Satellite-Systems.html)中已经介绍了），**频谱感应**允许次用户临时共享频段。基于机器学习的**认知无线电**算法还可以处理多目标问题等。然而这些方法通常只考虑不到五个可适应的无线电发射机参数和通信目标，并假设性能功能独立于通信信道，即操作环境。最流行的基于机器学习的算法之一的遗传算法，由于以批处理模式运行，因此对在线学习需求有一定限制。有人描述了分散频谱分配代理的分布式解决方案，其中强化学习被人工神经网络增强，该网络仅使用一个性能函数作为多用户相同资源的输入与输出值（对不起这里翻译不出来）。

因此据作者所知，上述技术都没有解决与考虑为天基通信系统提供多种无线电资源分配和通信目标，考虑到在线学习以及与上述环境动态相关的多种适应性参数和性能函数。有人试图解决空间应用中认知无线电的学习问题，提出了一种基于强化学习的方案，混合人工神经网络的混合解决方案等。然而这些解决方案受限于以下假设

- 无衰落信道
- 离散固定大小的状态与动作空间
- 状态-动作状态是可以存储的

这些假设导致基于机器学习的解决方案的实际应用十分有限。根据载波频率、航天器轨道动力学、大气条件、空间气象条件和机载可用存储器，多目标资源分配是一个巨大的挑战，需要一种新的解决方案，以便在不可预测的条件下进行操作。

本文中提出了一个新的天基通信系统的**认知引擎**的设计，解决了上述的局限性。该认知引擎在试图**在动态变化的通信信道中实现多个冲突目标**时会自动选择多个无线电发射机设置。利用了前人介绍的强化学习结构，提出了一个称为`virtual exploration`的升级版本，并将其与一种新的深度神经网络的集成设计 和两种新算法结合而成，以实现**多目标强化学习**(`multi-objective reinforcement learning, MORL`)的开发部分（？）。由此，该认知引擎能够实现下述的所有目标

- 具有固定内存大小的无表的 状态-动作 映射（内存占用固定，相对前面的内存足够大）
- 在动态变化的信道上运行
- 状态与决策分离
- 使用连续的动作与状态空间

这些特点是以**处理需求来训练神经网络并用它们来预测**为代价实现的。总之神经网络的引入消除了强化学习中的状态-动作表和Q值，也是因此将状态与动作解耦，允许将一个动作映射到几个其他状态，从而实现对动态信道的操作。最终连续的空间导致了接近最优解决方案（这又是说了句什么玩意）。

第二部分简述了本项目中使用的机器学习概念，第三部分描述了解决方案，第四部分仿真，第五部分得出结论。

## II - 机器学习概述 - MACHINE LEARNING OVERVIEW

机器学习是一个用于描述自动执行计算决策任务的若干理论和算法的术语。与本文相关的，机器学习值得注意的两个研究进展为

- 2015年深度Q网络(`Deep Q-Network, DQN`)用于`Atari`游戏时有比人类的更优表现。
- 2016年`AlphaGo`赢得了围棋世界冠军

以上的两个决策系统基于的原理一直在推动机器学习在更多不同领域的研究。目前而言主要的驱动因素是计算机视觉系统，其应用主要在自动驾驶汽车等。

当然这些系统都是利用了本文中介绍的深度神经网络和强化学习方法的。最近的技术革命激发了使用以上控制概念进行卫星通信的提议，但仅管如此，应用时许多要求是完全不同的。据作者所知，这些要求导致了目前文献中没有的算法的研究和开发（废你妈话）。利用强化学习方法和深度神经网络基本原理，本文提出了混合方法设计，并对结果进行了仿真与讨论。

### A - 神经网络概述 - Neural Networks Overview

人工神经网络是一种将输入映射到输出（译者注，通常是非线性映射，因为采用了激活函数）的方法，通常用于模式识别(`pattern recognition`)问题或函数拟合(`function-ﬁtting`)问题。由三层甚至更多层神经元组成的网络通常被称为深度网络。在本文中，深度网络通过将**行动**(`action`)映射到**奖励**(`reward`)、将**状态**(`state`)映射到**行动**(`action`)，近似非线性的环境影响。

神经网络算法基本上由两部分步骤构成，即训练和预测。这是监督学习方法，首先将输入与输出对应的训练集输入进行训练，在损失函数下满足了一定的性能要求后，该网络就可以用来进行预测了。这里对该算法不做详细介绍。

目前的文献资料中没有关于如何选择神经网络的结构的说明，因此本文中选择了简单的全连接网络，其设计细节在后面会介绍。

### B - 强化学习概述 - Reinforcement Learning Overview

强化学习算法通过与环境交互反复试错的方式进行学习。基于预定目标，强化学习**智能体**`agent`会查找实现这些目标时优化系统性能的决策(`action`)。在传统的强化学习算法中，`agent`根据离散时刻$$k​$$的`exploration probability function`$$f(\varepsilon)​$$计算的`exploration probability value`$$\varepsilon_k​$$在`exploitation`与`action`之间进行交替选择。

> 即智能体根据一定概率进行其他决策的探索，否则就直接决策而不探索其他决策方式。

强化学习问题可以被建模为**状态转移问题**，而状态转移问题本身就可以被建模为马尔科夫决策过程(`Markov Decision Process, MDP`)。本文中，状态转移假定是确定性的，`action`是应用一组无线电发射机参数，`state`是一组相关的通信系统性能值。有关`action and state`的更多详细说明，请参阅后文。

通常控制问题需要计算将观察到的`state`映射到`action`的`policy`。本文介绍的方法设计控制无线电参数，以便在信道条件发生变化时，性能在整个时间内保持为最佳水平。此时的`environment`由卫星通信信道组成，主要影响因素为发射机与接收机的视线路径、其附近的环境（地面站附近的建筑物或航天器附近天线的结构）以及大气与空间天气动态。如果采用状态转换和`action-state`模型，就必须考虑这些高度复杂的动态过程中的所有变量。出于这个原因，假设这些模型由于过于复杂或难以获取(`obtain`)而被认为是（效果）未知的，难以达成的，并且考虑到冲突目标，期望平衡探索新行动和利用已知行为的学习方法建议`action`。

应该通过下式给出的贪婪策略`policy`，对状态$$s$$下可能的所有行动，评估 表示在遵循`policy`$$\pi$$时处于`state`$$s$$时所采取的特定`action`$$a$$的值的`action-value`函数$$Q_\pi(s,a)$$

$$\pi(s)=arg\max\limits_a(Q(s,a))\tag{1}​$$

对于每一个$$s\in S$$，`policy`都会选出一个$$Q(s,a)$$最大的$$a\in A$$。

在包含数千个`action`$$a$$的连续或离散的$$A$$的问题时，在`state`$$s$$评估所有$$Q(s,a)$$可能是不行的。这里是无线通信的情况，在无线通信中，探索每一个`action`$$a$$可能会花费很多时间，并且迫使通信系统经历较为严重的性能下跌。实际的替代方案是让`agent`保持探索的动作。

一些文献中提到，认知无线电对即时`reward`敏感，并且任何`action`可以从任意`state`采取，而不需要计划在线申请。这些假设导致`Bellman Q-value function`稍加修改

$$Q_{k+1}(s_k,a_k)=Q_k(s_k,a_k)+\alpha \left[ r_k-Q_k(s_k,a_k) \right]\tag{2}$$

$$Q_{k+1}$$是得到`reward`$$r_k$$之后更新的`Q-val`，根据参数可以知道，此处的条件仍然是`state`$$s_k$$与`action`$$a_k$$，$$\alpha$$是学习率。即使状态转移与`state-action`模型是未知的，$$f(\varepsilon)$$和`reward function`$$\rho$$，其应用如下，仍然需要被定义。

$$r_k=\rho(s_k,a_k)\tag{3}$$

`state-action`映射函数

$$a_k=h(s_k)\tag{4}$$

在算式$$(1)​$$中给出了。

### C - RLNN 概述 - RLNN Overview

已经有先人提出过**多目标强化学习**(`MORL`)算法。也已经有人率先将前人的结果融合创新，设计了`NN-based RL, RLNN`，例如有将两篇文献中强化学习与神经网络结合在一起的人，假定时不变`AWGN`同步卫星通信信道进行设计的。它使用神经网络来虚拟地测试不同`action`，并测量它们的性能，从而允许对环境进行“虚拟探索”。

训练后，`RLNN`预测所有`actions`的多目标效能分数（即不同目标的加权和）。接下来根据性能阈值来将`action`分类为好或坏，即无线电参数的设定是否合适，并将所有效能得分计算为最高效能的百分比。然后从任意一类中依`action rejection probability`选取一个`action`。图1说明了被RLNN探索算法拒绝的`action`的时间序列性能的一个示例，这些`action`被预测在低于0.7的阈值下执行。相反，它建议执行高于该阈值的动作。

<div style="width:50%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNFig1.png" alt="" >
</div>
> 一段50秒时间序列示例，图中是RLNN仅采用虚拟探索，并将拒绝概率设置为1。在前200个数据包时，神经网络收集训练数据。显然，我们设置的阈值0.7，则预测出来性能不高于0.7的`action`的**探索**都被拒绝了。避免了不利的探索。

强化学习神经网络的一个重要好处在于知道每个动作的预期性能值（先验）,从而允许它们被分类。另一个好处就是可以通过`action rejection probability`来控制探索好的或坏的`action`的数量，通过滤除坏的`action`来提高系统整体的性能，`agent`可以避免花费时间去探索那些预测出来性能表现不佳的`action`。

上述的算法的主要缺陷是没有考虑到动态信道的问题。下图展示了一个时序示例，其中信道被假设为静态信道，给出的是每个动作的多目标性能得分。根据轨道的动态变化，载波频率、大气或空间天气可能会导致信道出现快速或慢速衰落，从而使其产生动态变化。

<div style="width:60%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNFig2.png" alt="" >
</div>


> 强化学习算法在同步卫星通信信道上搜索可用的更好`action`，搜索前首先考虑上一次探索到的`action`性能表现值。从图中可以明显看出决策`action`在不同时间段内的最优决策方案，连续位置可以覆盖整个时间段，此处的`action`就是系统的最优决策。

使用过的每一个离散的`action`都将其得分值存储在`Q-vector`中（假设`reward`与`state`是相同的），这些值在探索期间接收新值并更新现有值。

但是如果信道是动态变化的，这种方法就有三种主要缺陷

- 现在采取某种`action`的性能值得分可能和过去或将来采取同一个`action`的不同（这不是废话么，`action`的得分本来就是受时间和环境影响的啊，硬加一个缺陷还行）

  - 之前计算的性能值在后来使用的时候就过时了，这将导致采用某种`action`的时候做出错误决策。此问题的解决方法就是更新对应的性能值。这对于非动态变化的信道而言似乎没有大问题，但是对于动态变化的信道，某个特定的`action`可能在不同的环境下有不同的性能表现，之前探索计算的性能得分就无效了。下图很好的说明了这一点。该图片描述了两个不同的`action`随信道变化的仿真性能。即使是对于不同的通信任务，有的`action`也会表现出微小的变化，但是有的`action`随时间推移而产生的性能变化是剧烈非线性的。下方左右两图的剧烈反差实际上推动了本次研究。

- 必须为每个不同的多维`state`储存特定`action`的性能表现值，其中`state`是由连续变量表示的

  - 该缺陷导致需要大量的存储器（指数增加的存储器），这是由于当假设信道条件离散改变时，每个动作表现为不同性能水平。实际上状态是连续变化的，原计算需要更改如下

    $$Q_{k+1}(s(t),a_k)=Q_{k}(s(t),a_k)+\alpha[r_k(t)-Q_{k}(s(t),a_k)]\tag{5}$$

    这就表示给定`action`的得分随时间的动态行为。

- 必须为动态变化的特定信道条件存储每个`action-state`的性能表现值，但是信道变化也是连续的。

  - 该缺陷 增加了性能值的另一个维度，即随时间的变化，让数据的存储变的更为困难不可实现。从在线系统操作的实现角度来看，保存所有信息是不切实际而且不可能的。

<div style="width:100%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNFig3.png" alt="" >
</div>

> 两个不同动作$$a_A$$和$$a_B$$的多目标性能分别如图所示。在环境变化的整个过程中，保持`action`恒定，遵循的`SNR`配置也在图中显示。$$a_A$$在不同任务的表现随时间的变化并不明显，但是$$a_B​$$的表现显然发生了明显的突变。

`2019-2-9 22:31:29`

本段还有很多内容没有完全看明白，还需要打磨一下。


## III - 提出解决方案 - PROPOSED SOLUTION

在继续提出算法之前，需要提到两个术语的区别，即强化学习的`state`和环境的`state level`。强化学习的`state`是`agent`进行观测计算的系统性能表现水平，它可以响应（反映？）强化学习中某个`action`的实施和现在的环境状态。在下一部分会说明，强化学习`state`是一个通信系统`fitness function`的一个特征。

> 之前把所有的fitness score 都翻译成了得分，主要是词穷，中文不好。后面基本上就不翻译了，有空把前面的也改了。

这里仿真的卫星通信中，我们假设信道发生的所有变化都能被传达（~~都能被届到~~），使用接收机处的`SNR`作为评价标准。也正是这样，`environment state`也被称为信道条件，信道状态等。

<div style="width:75%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNFig4.png" alt="" >
</div>

> 用于强化学习的探索(`exploration`)过程的深度网络，收到相同的多维输入进行输入。这里需要并行训练若干神经网络，并将其所有的预测输出平均为单个多目标性能值(`multi-objective performance value.`)。

对于动态变化的环境，强化学习神经网络现在要考虑到输入的$$\frac{E_s}{N_0}$$，这一部分是`Exploration NN`（如图）。该部分实际上是由相同训练数据集训练而成的几个深度神经网络`DNN`的集合，用于在相同输入的条件下生成多目标性能值。这些输出经过求平均后生成最终的预测值，也是如图所示。关于用于探索的网络结构，是采用`LM Algorithm`进行训练的前馈网络。这个`LM Algorithm`我有空写一篇文章出来说明。具体结构是三层全连接神经网络，只有`weight`而没有`bias`。两个隐藏层各包含`7`个和`50`个神经元，因此有$$449$$个参数（这里的想法是，输入7个神经元，然后过一个7的隐藏层，一个50的隐藏层，最后一层是一个输出神经元，$$7\times 7+7\times 50+50\times 1=449$$，这里感谢[@GeneZC](https://github.com/GeneZC)提供的帮助，哭了我这个不开窍的脑子），每层都采用`log-sigmoid`传递函数进行激活，输出层只有一个神经元，采用标准的线性映射函数。对于误差函数，此处采用经典的均方误差(`mean-square error`)作为性能的误差指标，其停止训练的条件有两个，

- 最小误差的梯度达到$$10^{-12}$$，这个很苛刻了，一般来说这么小的误差很有可能发生过拟合了。
- 最大验证校验达到$$20$$，这个也蛮苛刻的，`MATLAB`给出的默认值是6，即`validation set`连续迭代6次后误差不再下降，训练终止。这里设置为20，即迭代20次后不下降。过大的值一样容易导致过拟合。

训练的时候，数据集被**随机**分为$$7:3$$两部分，常见做法。$$70\%$$将用于进行训练(`training set`)，剩余的$$30\%$$还要对半分，平均分为两部分，$$15\%$$的部分用于测试(`testing set`)，另外的$$15\%$$的部分用于验证(`validation set`)。对应部分的用法在《统计学习方法》这本书里面有详细的说明，另外训练时的数据全部都按比例缩放到$$[-1,1]​$$范围内（这个是`scaling`操作，在[很久前的文章](https://psycholsc.github.io/notes/2018/08/10/CS229-Machine-Learning.html)里有介绍）。这个网络集合中有20个相同的网络结构，均采用相同的数据集进行训练，其输出结果用来对不同的`action`进行分类，分类依据是`action select agent`提供的性能表现阈值。另有`action rejection probability`控制从好的或坏的集合中随机选取动作，该动作即无线电参数的选择。

我们自己提出的混合`MORL Algorithm`（文中自称为先人`RLNN`的改良版`RLNN2`）采用两种不同的神经网络，一组用于`exploration`，另一组用于`exploitation`，其结构如下。

<div style="width:75%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNFig5.png" alt="" >
</div>

> 新型的为动态卫星通讯信道设计的`MORL`算法结构。此处我们提出的`RLNN2`和同样我们提出的探索与开发 神经网络 相互作用。

该结构将前面的神经网络和我们自己的新算法(`Exploitation NN`)进行了结合。这种新的算法可以处理卫星通信信道内的动态变化的衰减水平。该算法(`RLNN2`)的`agent`与环境进行交互，或者(`either`)通过虚拟探索`virtual exploration`探索不同的`action`（这将防止通信系统花费额外的时间探索可能性能不佳的参数组合），又或者(`or`)直接尝试曾经已经试过的`action`，为当前变化的信道预测最优的`action`（作者这里起了一个很中二很傻逼的名字，多维动作预测器，`multi-dimensional action predictor`）。

`Exploiting Reinforced Multi-Dimensional Actions:`为了解决前面第II部分提出的限制，我们删掉了`Q-value`计算的过程，而是到当前的环境状态水平（就是通信信道条件），建议使用另一个神经网络来预测应该使用哪个`action`。提出的这个神经网络的特征应该包括

- 解决使用过时的性能指标来决定使用哪个`action`的问题
- 不需要存储所有环境条件下所有动作状态的性能值
- 允许强化学习`state`与`action`相互分离（解耦、不相关等），这将允许在环境动态变化时通过开发不同的`action`来达到与之前相同的性能水平。

因此，我们提出的`NN2`结构是由一组神经网络阵列构成的，阵列中的每一个网络用于预测多维`action`中的一个维度。所有的`exploitation NN`具有相同的浅层结构，并接受相同的多维性能指标作为输入，如下图所示

<div style="width:75%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNFig6.png" alt="" >
</div>

该网络的输出向量是$$m=10$$个神经网络的平均值。该体系结构的选择是基于100次模拟运行中表现出最小`MSE`的结构，其性能会在下文简要分析。每个神经网络由两个全连接层组成，同样只有`weight`没有`bias`。隐藏层包含20个神经元（因此是160个参数，$$7\times 20+20\times 1=160$$），同样采用`log-sigmoid`传递函数进行激活，输出层是线性函数。基本参数实际上和前一个网络别无二致，只是结构稍加改变。

决定`NN2`输入的逻辑（即多目标性能值）十分复杂，需要很多条件语句。下面的算法1(`Algorithm 1`)描述了混合结构（即`RLNN2`）的一般操作过程，包括强化学习部分以及全连接网络之间的交互。`Algprithm 2`描述了`NN2`的输入选择逻辑，这是`Algorithm 1`中所必需的。所需的参数均在第`IV`部分进行定义和解释。

`2019-2-10 23:40:48`

说的什么几把垃圾

## IV - 结果 - RESULTS

在展示混合解决方案`RLNN2`结果之前，这里简单对一些术语和参数的仿真进行说明定义。（这说的是人话吗）

为了使系统仿真符合`DVB-S2`标准，可调参数，例如调制方案和编码速率，被假设为被**卫星发射机**和**地面接收机**使用。其他一些参数也被认为是可调的，它们的范围在表格中进行描述。

<div style="width:75%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNTab1.png" alt="" >
</div>

根据下述的`Algorithm 1`描述的顺序，发射机在动态信道环境中使用的`action`描述法是六个参数构成的，$$\overline{a}=(R_S,E_S,k,\beta,M,c)$$。

> 本文假设卫星发射机放大器工作在接近线性的区域
>
> 不同的调制方案采用不同的编码速率范围

对于下面给出的仿真结果，假设有一个慢衰落的通信信道，其衰减时间序列如第三张图所示，假设是用于晴空的条件。这个配置文件是从`STK`轨道模拟器获得的路径损耗的一个示例（一颗近地轨道卫星在一个类似于国际空间站的轨道上运行时）。这表示了发射信号由于卫星飞行轨迹而经历的自然环境衰减。

最佳动作适应是此处提出的`RLNN2`的最终目标。由于神经网络的训练并不是完美的，因此神经网络的输出只能代表尽可能接近最佳`action`的效果，这是考虑到训练缓冲区用于训练的数据是有限的。这些调整通过通信系统在强化学习算法的`exploration`或`exploitation`阶段期间选择不同的`action`而发生。在整个传输过程中，强化学习`agent`寻求优化冲突多目标函数组成的`fitness function score`。通信任务目标是由**六**个指标组成的向量，误码率(`BER`)，吞吐量(`Thrp`)，带宽(`BW`)，频谱效率(`Spc_eff`)，额外消耗功率(`Pwr_con`)和功率效率(`Pwr_eff`)，所有值都被缩放到$$[0,1]$$的范围内。假设额外消耗功率是加到恒定功率上的可变功率量，**后面这句话不知道在说啥**。

出于概念验证的目的，本文的方法中考虑的多目标是由**吞吐量函数**、**带宽函数**、**频谱效率函数**、**功率效率函数**、**消耗的额外功率函数**组成，其计算如下

$$f_{Thrp}=R_Skc\tag{6}$$

$$f_{BW}=R_S(1+\beta)\tag{7}$$

$$f_{Spc\_eff}=\frac{kc}{1+\beta}\tag{8}$$

$$f_{Pwr\_eff}=\frac{kc}{10^{\frac{E_s}{10N_0}}R_S}\tag{9}$$

$$f_{Pwr\_con}=E_S\cdot R_S\tag{10}$$

通过`MATLAB`中对`DVB-S2`通信系统的仿真获得的曲线函数进行插值计算误码率（这里是真神奇，连中国本科生都会的仿真居然还有人专门写在论文里要怎么用。绝了。）。假设帧长度为`64800 bits`，即`DVB-S2`中的长帧类型。

先前定义的`reward function`，即`fitness score`，计算函数如下。实际上是一个加权过程，代表该通信过程中对哪项指标更为重视。


$$
\begin{equation*}
\begin{split}
f_{obs}(x) =& w_1f_{Thrp}+w_2f_{BER}+w_3f_{BW}+\\ & w_4f_{Spc\_eff}+w_5f_{Pwr\_eff}+w_6f_{Pwr\_con}

\end{split}
\tag{11}
\end{equation*}
$$

其中权重在下表给出，是用户自己可以决定的，$$x​$$是对所有参数的一个汇总。

<div style="width:75%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNTab2.png" alt="" >
</div>
---

`Algorithm 1`定义了神经网络训练缓冲区(`buffer`)的使用，该缓冲区总共包含$$NN_{bs}​$$个条目，每个条目由`action(a)`、`state(s)`和`fitness function value`，这些都是在`exploration`过程中加入的。当然无论什么时候对这个`buffer`被填满，神经网络就要经历一轮新的训练，此时还会将最旧的一组条目从`buffer`中删除，其大小为$$NN_{dump}​$$。总而言之就是，每填充$$NN_{bs}​$$个条目后，神经网络训练一遍，并且丢弃最旧的$$NN_{dump}​$$个条目等待重新填充，以保持一定的时效性，并同时提高效率降低运算量。

在每个数据包进行传输之前，强化学习`agent`通过$$f(\varepsilon)$$计算一个$$\varepsilon_k$$来决定是进行`exploration`还是`exploitation`。如果随机数$$z$$小于$$\varepsilon_k$$，则`agent`通过`NN1`进行“虚拟探索”，提出一个新的随机值`u`，并将其与`action rejection value`$$tr_a$$进行对比，以便从好组或坏组选择一个`action`。好组与坏组的定义是性能表现阈值，该阈值是虚拟探索时该项的`fitness score`与当前最大的`fitness score`的比值，即占比，这里的阈值是人为规定的，$$min_{good\%}$$。

对于下面给出的仿真结果来看，假设值定义如下

- $$NN_{bs}=200$$，即缓冲区容量定义为$$200$$个条目
- $$NN_{dump}=50$$，即每训练一次，丢掉$$50$$个等待下次填充
- $$tr_a=0.95​$$，`action`拒绝概率
- $$min_{good\%}=0.9$$，达到这个水平才是好的
- $$f(\varepsilon)​$$是随时间变化的，假定$$\varepsilon_k=\frac{1}{k}​$$，当取值为$$10^{-4}​$$时进行重置

---

`Algorithm 2`定义了`NN2`的输入选择逻辑，主要为两个机制对输入进行选择，慢恢复与快恢复。前者涉及对存储在神经网络训练缓冲区的性能表现值的利用，该缓冲区按`fitness score`进行降序排列；后者重置神经网络训练缓冲区，这将导致在`forced exploration mode`下收集足够的数据之后对两个神经网络进行重新训练。由于快恢复强制在$$NN_{bs}$$数据包的持续时间内进行连续探索，因此触发时通过$$m_{reset}$$进行控制。这算是一个超参数之类的东西，因为它对于所有任务而言目前均采用$$m_{reset}=0.5$$进行控制。

---

第二个表格中的所有任务就是本次仿真时的所有任务了。由于每一次任务每次迭代所需的持续时间问题，本次结果采用$$10$$次平均。

> MATLAB仿真每次4-8小时，使用两个Intel Xeon 16核2.3GHz处理器总共运算了320小时。

下图提供了每个任务的`fitness score`的时间序列示例，其中`LEO satellite`低轨卫星通过地面站的持续时间如前文所示为`512 s`。这些事件序列包括了`exploration`和`explitation`过程的性能表现值。

### A - RLNN2的性能评价 - Remarks on RLNN2 Performance

如图所示

<div style="width:75%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNFig8.png" alt="" >
</div>

图像a-f分别呈现的是任务1-6的归一化得分分布。尽管每一个任务都有不同的配置，但是所有的图像中我们都能看出，性能表现均集中在较高水平。因为我们的目标是让更多数的数据包发送后的得分尽可能地接近理想分数，通过穷举搜索评估没个时间瞬间的所有动作性能。这些图像显示`RLNN2`能选择好的`action`，在信道动态变化时能够学习`reward`和`action`的关系。

为了更好理解该方案的准确性，用`1`减去该结果得到归一化误差。归一化误差是每一个数据包传递时的实际得分与理想的分的差值，用`1`减去这个结果后得到下图

<div style="width:75%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNFig9.png" alt="" >
</div>
得到的精确度分布曲线如图9所示，预计将作为未来空间通信系统CE研究的基准。 对于所有任务，`RLNN2`算法可以使大多数数据包根据需要集中在非常高的精度值附近。

另一种关于误差的度量方式是计算上两个图中的曲线保卫面积，计算结果如下

<div style="width:75%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNTab3.png" alt="" >
</div>

在性能表现方面，任务4的积分最小为`0.72`，任务6的积分最大为`0.88`。就准确度而言，误差分布的积分在任务3中有最小值为`0.01`，在任务2中有最大值为`0.06`。由于每个任务评价`fitness score`时的权重不同，因此其理想得分也有所差异，所以误差成为了信道动态变化时更好的度量方式。

表格里面同样展示了当`NN1`和`NN2`的容量仅为`1`时的误差水平。这里的数值选择直接影响训练时间和预测决策的时间。即使对于`MATLAB`中的仿真过程，整个时间缩短了`12`倍，而且还是没有专门的硬件实现的基础上。不过这些就超出了本次讨论的定义了。

不过对于所有任务来说，即使选择了更大的`NN`容量，其结果都是类似的，看起来误差成倍数增长实际上都在一个较小的水平。此外在相同的信道条件和任务配置的条件下，前面定义的得分值函数可能在不同`action`的条件下达到相同的结果。这其实可以允许我们在灵活选择成本的条件与多目标性能下进行不同的选择。

### B - RLNN2的性能表现权衡 - RLNN2 Performance Trade-Off

仿真结果验证了我们提出的解决方案是符合穷举搜索结果的上限的，因为结构从未达到过超越理想的性能上限。除了第`III`部分中激发`NN2`提出的若干限制以外，使用神经网络来选择要被利用的`action`提供了更多的灵活性而非性能提升。例如仅对于一个特征是配的系统，例如对于`DVB-S2`标准使用的自适应协议而言，并不能同时优化整组无线电资源。

另外`NN1`可以再强化学习的`Exploration`阶段实现性能的控制。假如`state - action`映射的粒度足够高（假设内存足够存储），并且$$\varepsilon_k$$也足够高时，我们可以控制系统的性能更高，但是要消耗更多的资源与时间。

使用神经网络进行训练和预测提供了灵活性，甚至可以允许同时选择更多的参数，同时即使允许使用连续型的`state - action`映射也能节约内存的使用，是一种比较现实的解决方案。同时本方法还是可以采用固定内存方法实现的，因为参数的数量是严格固定的；本方式还可以不使用映射表方法。不过未来的通信系统可能会使用类似的算法去实现其他性能或任务。

*后面都是自卖自夸的客套话我实在是不愿意往上码了，自夸完他还要说一句 “因为以前没人做这么多目标的优化，因此我们之间相互比较是不公平的，毕竟我们太强了” 这样的话*

*还顺便贬低了一波遗传算法，因为计算一轮的延迟太高了，顺手自夸一句自己是理想解决方案，还要说自己是未来的基准，我佛了。*




## V - 结论 - CONCLUSIONS

本文提出的算法构成了认知引擎的核心，可以作为下一代航天器和卫星之间空间通信系统的发展基线。提出了一种基于强化学习的多目标性能表现的概念验证方法。他依赖于在动态变化的信道中使用神经网络进行`exploiting`和`exploring`无线电参数集合。

所提出的解决方案包含一个深度网络的合集，用于强化学习探索阶段提供性能控制的`action`参数的探索；一个浅层网络阵列合集，用于预测某个已知性能目标的无线电参数。

所提出的算法使得能够在连续`action - state`空间中灵活地进行若干无线电参数的自适应，并不是改变特定的一个性能度量指标。同时保持了随时间变化对存储器使用的要求的稳定性。系统的整体性能是在基于系统操作员定义的所选通信任务配置的条件下进行监测的。

另外还提出了该解决方案的计算成本、执行时间和性能准确性的权衡分析。仿真结果表明，与理想的解决方案相比，性能误差非常低。所有的任务中实现准确率均高于`80%`，这些有望成为未来空间通信系统认知引擎的研究基准（我觉得不行）。



## EX - 鸣谢&后记

- 红豆泥感谢[@GeneZC](https://github.com/GeneZC)佬，我不会的问题都能帮我解决，隔着屏幕就感觉到了一股大佬的气息！
- 这篇文章的逻辑之混乱真的是常人难以理解的。我看完了一遍以后仍然不知道他操作的流程是什么样的的，胡说八道的内容也很多。
- `2019-2-12 10:58:53`计划着开始写个程序仿真一下看看了。

首先是算法1，其流程如下

<div style="width:50%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNAlg1.png" alt="" >
</div>

在算法进行之前，需要初始化两个神经网络1和2，即进行模型的建立，资源分配和初始化赋值。后面的若干个参数分别为

- $$R_S:​$$符号速率
- $$E_S:​$$每个符号的能量
- $$k:$$每个符号中含多少`bit`信息
- $$\beta:​$$滚降系数，成形滤波器使用
- $$M:$$调制阶数
- $$c:$$编码率（信息率），即信息占所有编码结果的比例。
- $$NN_{bs}:​$$神经网络训练集的大小。$$NN_{bs}=200​$$，即缓冲区容量定义为$$200​$$个条目
- $$NN_{dump}:​$$神经网络中缓冲区大小，文中有具体介绍作用，$$NN_{dump}=50​$$，即每训练一次，丢掉$$50​$$个等待下次填充
- $$f(\varepsilon):$$描述$$\varepsilon_k$$的函数，假定$$\varepsilon_k=\frac{1}{k}$$，当取值为$$10^{-4}$$时进行重置
- $$tr_a:$$`action`拒绝概率，$$tr_a=0.95$$
- $$min_{good\%}:$$性能指标规划，取$$0.9$$

注意其实每一个参数都有取值范围，我们此时生成一个集合，用来储存所有取值的组合。该全集我们称为`U`

整个算法1现在就开始了，实际上算法1整个流程就是一个大循环，`loop`，现在做完了初始化，流程正式开始（进入循环）

- 如果我们的取值范围发生了改变，那么立即更新全集`U`
- 检查神经网络的训练数据缓冲区是否满了，即拥有200组数据供训练
  - 如果缓冲区没有达到200个，就开始强制探索模式，`exploration only`
- $$z:=random(0,1)​$$，均匀分布的随机数
- $$z$$和$$\varepsilon_k$$进行比较
  - 如果$$z$$比较小，则采用`NN1`预测`action`
  - 使用$$min_{good\%}$$进行分类，高于该值认为是`good`，否则认为是`bad`
  - $$u:=random(0,1)$$，均匀分布的随机数
  - 比较$$u$$和$$tr_a$$
    - 如果$$u$$更小，此时选择`good set`中的`action`
    - 如果$$u$$更大，此时选择`bad set`中的`action`
  - 以上为$$z$$较小的条件
  - 如果$$z$$比较大，则采用`NN2`预测被利用的`action a `
  - 执行`action a`
  - 测量（或者说读取）强化学习`state s`
  - 计算`multi-objective fitness function`$$f_{obs}(x)$$
  - 通过算法2选择一个`NN2`的输入
  - 更新神经网络训练缓冲区
  - 构造`NN1`和`NN2`的训练集
  - 如果 训练数据缓冲区达到了`200`组，则丢弃`50`个条目

回到`loop`循环首

以上是算法1的流程。

---

下面是算法2的相关内容，其流程如下

<div style="width:50%; margin-left:auto; margin-right:auto; margin-bottom:8px; margin-top:8px;">
<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLNNAlg2.png" alt="" >
</div>

首先是初始化取值，这里给出的是$$max\_f_{obs}=0$$，$$m_{reset}$$和$$Exploit_{score}=[]$$

- 检查训练数据缓冲区是否满
  - 若没满，则循环执行以下操作
  - 如果是强制探索模式，则$$Exploit_{score}=max(f_{obs}(x))$$
  - 若缓存区满了，该`while`段结束。
- 如果 $$f_{obs}>max\_f_{obs}​$$
  - $$max\_f_{obs}=f_{obs}(x)$$
  - 如果 $$z<\varepsilon_k$$
    - $$NN2_{input}=state$$
  - 否则
    - 如果$$f_{obs}<Exploit_{score}​$$
      - 如果$$Exploit_{score}-f_{obs}>m_{reset}$$且连续两个时间步内`action`相同$$a_k=a_{k-1}$$
        - 重置神经网络训练缓冲区
        - 开始强制探索模式
      - 否则如果$$f_{obs}<0.9\times Exploit_{score}​$$
        - $$NN2_{input}$$从训练缓冲区内接收`state`
      - 否则如果$$f_{obs}>0.9\times Exploit_{score}​$$
        - $$Exploit_{score}=f_{obs}(x)​$$
      - 否则
        - $$NN2_{input}=last\_NN2_{input}​$$
    - 否则
      - $$Exploit_{score}=f_{obs}(x)$$
      - $$last\_NN2_{input}=state$$

复杂的关系。









