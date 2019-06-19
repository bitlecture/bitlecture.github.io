---
layout: post
comments: true
title:  "Reinforcement Learning: The Multi-Armed Bandit"
excerpt: "-"
date:   2019-02-24 14:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>


---

很久没更过了。2018年下半年过得很狗屎。十分难受。

`2019-3-1 19:43:00`

读这本书真的是有感而发，国外的教授写的书水平确实比较高，虽然看起来冗余内容比较多，但是仔细读一遍还是觉得这本书质量颇高，解释的很清晰，而且干净利落条理清楚，英文用词准确还不会过分复杂，内容高度自洽。

真的强啊。

# Reinforcement Learning

`Reinforcement Learning`的正统翻译目前是**强化学习**，是所谓的机器学习的一个分支领域，强调如何基于环境而行动，以获取最大化的预期利益。其灵感来自于心理学中的行为主义理论，即有机体如何在环境基于的奖励或惩罚的刺激下，逐步形成对刺激的预期，产生能获得最大利益的习惯性行为。这个方法具有普适性，因此在其他许多领域都有研究，例如博弈论、控制论、运筹学、信息论、仿真优化等。在一些研究环境下，RL还被称为近似动态规划(`approximate dynamic programming, ADP`)。在最优控制理论中也有研究，主要研究方向为最优解的存在与特性，而非此学科的学习、近似等。该理论还被解释为在有限理性的条件下如何平衡。

在机器学习问题上，环境通常被规范为Markov决策过程，所以许多强化学习算法在这种条件下使用动态规划技巧。强化学习不需要对Markov决策过程的知识。强化学习和标准的监督式学习的区别在于不需要出现正确的输入输出对，也不需要精确校正次优化的行为。强化学习更加专注于在线规划，需要在**探索**和**遵从**之间找到平衡。

>Reinforcement is about taking suitable action to maximize reward in a particular situation.

由于不像传统强化学习一样是通过正确输入输出来进行训练的，而是通过一个`agent`决策在指定任务下应怎么做。在没有数据基础的前提下，强化学习是通过自己的经验去学习的。

强化学习是一个很大的坑，从头写写看，看看能坚持多久吧。

## 第一章 Introduction

### 引入

当我们说起学习的本质的时候，我们总会认为第一个想到，我们是通过与环境的互动学习的。我们出生的时候，并没有老师教我们如何进行活动，但是我们自己确实与环境有直接的感觉运动联系。人类的一生中，许多学习都是这样的，就像今天`2019-1-23`，面对两把外观相同的钥匙，经过一段时间的观察，我就发现哪一把是刚配的，哪一把是旧的。

在我们的一生中，通过与环境的互动来进行学习的例子很多，我们大多数时间都是通过环境的反应进行学习的，开车，交谈，一切都是这样，我们敏锐地意识到我们的环境是如何对我们所做的做出反应的，我们寻求通过我们的行为来影响所发生的事情。从互动中学习是几乎所有学习和智力理论的基础。

现在我们尝试找出一种计算方法，让我们从互动中进行学习。后面的探索基本上也都是基于理想的学习情境，站在研究者的角度评价不同学习策略的有效性。

强化学习是学习如何决策的，如何将情境映射到决策，从而最大化某种数字化的奖励信号（被称为回报，`reward`）。学习者并没有被告知应该如何决策，而是通过尝试来发现哪一种决策可以获得最大的收益。在复杂而有挑战性的情境中，行动可能不仅会立即影响回报收益，还有可能影响下一种情况，并通过这种情况进而影响后续的回报。

书中利用动力学系统理论的思想，特别是作为不完全已知马尔可夫决策过程的最优控制，将强化学习问题形式化，基本想法是简单地捕获`agent`随时间与环境交互以实现目标所面临的实际问题的最重要方面。

`agent`必须能够在某种程度上感知环境的状态，并且必须能够采取影响状态的行为。

`agent`还必须有一个或多个与环境状态相关的目标。

`Markov`决策过程就是以包含这三个层次为目标的，即**感知**、**决策**和**目标**。仅仅对于它们可能的最简单的形式，但是绝对不会将任何一个看的不重要。任何一个非常适合用来解决这些问题的方法都可以被认为是强化学习方法。

**监督学习**是通过`training set`和`labeled data`进行学习的，通常是分类器或回归问题；典型的**非监督学习**是发现`unlabeled data`中的隐藏的一些信息结构。这两个分类看似对机器学习的范式进行了具体的分类，但是并没有。强化学习并不属于典型的非监督学习，但是他并不是用于发现隐藏于信息中的内容的，而是希望让某种`reward`最大化，仅仅发现信息结构并不能完全达到这个目的，虽然是有益的。因此可以认为强化学习是机器学习的第三个门类，另外值得一提的是，机器学习的门类可能远不止目前的三个门类。

强化学习遇到的其中一个重要问题是，如何在`exploration`和`exploitation`之间寻求平衡。

- 为了获得更多的`reward`，`agent`自然会优先选择曾经尝试过的并产生了大量`reward`的决策
- 上述决策的基础是，必须曾尝试过这个没有选择过的决策

因此`agent`必须要利用其已有的经验来获得`reward`，但也必须进行探索，以便在未来做出更好的行动选择。`agent`必须尝试各种决策，并逐步支持那些看起来最好的行为决策，与此同时也要继续对其他决策进行探索。在一个随机任务中，每一个行为都需要经历多次尝试，以获得随机行为的可靠估计。数十年前数学家就对这个所谓的`exploration–exploitation dilemma`进行过研究，然而并没有什么可靠的解决方案。上述的两种机器学习方法也都还没解决这个问题。

强化学习的另一个关键特征是它明确地考虑了目标导向的`agent`与不确定环境相互作用的整个问题。这与很多只考虑子问题而不考虑该如何将它们在更大范围内实现的方法形成对比，例如监督学习方法。这种只关心子问题的方法存在明显限制，例如它们只学到了如何对指定数据集进行回归，而不知道面对回归问题时该如何进行操作。（不知道这个理解对不对）（这个**子问题（Subproblems）**实在是不懂在说啥）

目前强化学习与许多其他工程和科学学科已经有了富有成效的结合，也反过来促进心理学与神经科学的发展。这些内容在书中最后部分。

至于发现机器智能的一般原则的过程中，强化学习所做的贡献也是十分有限的，目前投入的研究资源也很少，在这条道路上还得不出结论。

> Modern artiﬁcial intelligence now includes much research looking for general principles of learning, search, and decision making, as well as trying to incorporate vast amounts of domain knowledge. It is not clear how far back the pendulum will swing, but reinforcement learning research is certainly part of the swing back toward simpler and fewer general principles of artiﬁcial intelligence.

简单举例几个强化学习应用以后我就直接进入主题。

- 象棋高手的每一步决策
- 自适应控制器实时调整某项参数
- 新生动物在出生后的几分钟内学会站立（站起来！萌萌！站起来！）
- 扫地机器人基于当前电量决策继续进行垃圾收集还是还是回到充电站。

这些都涉及一个积极的决策`agent`与其环境的相互作用，尽管环境可能并不确定，但是`agent`仍然寻求一个目标。`agent`的行为决策可能会影响未来的环境，例如上述的象棋决策，从而会进一步影响`agent`以后可能获得的选择与机会。

>Correct choice requires taking into account indirect, delayed consequences of actions, and thus may require foresight or planning.

### 基本元素

除了`agent`和`environment`，我们还可以识别强化学习系统的四个主要基本元素，策略`policy`，奖励信号`reward signal`，价值函数`value function`，（可选）环境模型`model`

- `policy`定义了`agent`在指定时间的行为方式，即从环境感知到的状态与决策之间的映射关系。一般来说，`policy`可能是随机的
- `reward`是强化学习问题的目标。在每个时间步骤中，环境都向`agent`发送一个数值，称为`reward`。`agent`就是在长远角度来看尽可能地获取最大的`total reward`。一般来说，奖励信号可能是环境状态和所采取行动的随机函数。
- `value function`指定的是`agent`期望的总`reward`。`reward`可以说是短期的收益，而`value`才是长期收益。
    - 从某种意义上说，`reward`是首要的，而`value`会次要一些。但是我们实际采取决策的时候，`value`才是最重要的因素，因为我们的目标是长远来看获取最大收益。不过确定`value`比确定`reward`要困难得多，对于`value`的估计也是强化学习算法中最重要的部分，过去数十年的研究也是基于此。
- `environment model`是对环境行为进行模拟的模型。具体内容在第8章才涉及



### 限制

强化学习过于依赖`state`这个概念。在RNN中`state`是灵魂，强化学习本不应是类似的模型，但是却一样地依赖于状态。强化学习中的每一个元素都可以看做是一个状态。我们主要关心的问题不是如何设计状态信号，而是在任意状态信号可用时该如何采取行动。



**这里本该有一个Tic-Tac-Toe的示例，此处先行略去**

## 第二章 Multi-armed Bandit Problem

`2019-2-28 15:30:06`重写

前面提到的强化学习有别于**监督学习方法**和**非监督学习方法**，其主要区别在于，强化学习通过训练数据、信息去评价每一次`action`，而非通过正确标注的输入输出来指导系统的改善。但是也正是如此，我们需要积极的`exploration`，即明确的对正确`action`的探索。这里我们先简单介绍两个`feedback`，一个是`evaluation feedback`和`instruction feedback`。前者可以直译为“**评价反馈**”，是完全依赖于`action`的一种反馈，真实反馈目前`action`的水平；后者是“**指导反馈**”，完全独立于采取的`action`，反馈的是当前环境水平。目前的监督学习方法实际上使用的是这类反馈。

本节通过最简单的多臂赌博机问题，介绍最简单的强化学习模型，并且在此基础上理解上述两类反馈的思想，结合两类反馈进行实现。

### K - Armed Bandit

K臂赌博机问题实际上是一个循环决策问题，每个时间步面对一个`K-option`问题，每个选项都依一定的概率分布返回一个`reward`。该问题的目标就是最大化累计`reward`，是一个决策问题。

对于最简单的赌博机模型，即`Bernoulli Bandit`模型，赌博机会根据预先设定的概率返回`reward`为$$1$$，否则返回$$0$$。但是这里并没有采用这种简单的情况，而是采用依正态分布的方式返回`reward`。一方面能让问题稍显复杂，也能更加接近实际的决策问题。

最简单的赌博机模型如下

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/bern_bandit.png" style="display: inline-block;" width="500"/>
</div>

这个模型可以类比很多现实的决策问题，例如赌博问题，医生对病人治疗方案选择问题等。对本模型的介绍，就从`bandit`问题开始。在这个问题中，我们每一个时间步都会做一个唯一的决策`action`（即选择一台赌博机），每一个`action`都会得到一个对应的`reward`（赌博机给出收益）。我们用$$A_t,R_t$$分别表示$$t$$时刻的`action`和`reward`。对于每一台赌博机，我们**首先**都认为存在一个固定的期望收益（这种情况被称为`stationary`情况，大多数强化学习遇到的问题实际上都是`non-stationary`问题，这里只是最简化模型），用$$q_*(a)$$来表示，
$$
\begin{equation*}
\begin{split}
q_*(a)=\mathbb{E}[R_t\mid A_t=a]
\end{split}
\tag{1}
\end{equation*}
$$
即每一个`action`的收益期望。

假如我们知道了每个行动的期望收益，那我们的决策就变得简单了——只需要选择期望收益最大的那个决策，在足够长的时间后一定是这个选项带来的平均收益最大（逼近期望值）。

可是对于一般的决策问题，我们是肯定对这个数值一无所知的。我们一开始对各台赌博机没有任何先验知识，那么只能通过我们的行为收集环境信息，以此来估计各个赌博机的收益期望。这个估计的结果我们标记为$$\hat Q_t(a)$$，即时间步$$t$$时我们对各个选项的收益期望的估计。当然为了长期收益达到最高，我们希望这个估计值能够尽可能得逼近真实值。

如果我们持续对$$q$$进行估计，那么每个时刻总是至少有一个$$Q_t(a)$$值是最大的，这个值我们称为`greedy value`，一般的我们会根据$$\hat Q$$进行决策，这种决策方法就是`greedy`决策，对应的行动为`greedy action`。当我们依据这个策略进行决策的时候，我们就被称为 进行`exploit`。而相反的，如果我们不采用`greedy action`的话，就被称为`explore`。

那么到此为止，我们已经发现了一个最为重要的问题，即**Exploration vs Exploitation**的问题。

### Exploration vs Exploitation

如上方所说，这两个不同类型的`action`是需要我们进行选择的，因为如果我们持续`exploit`，我们就只会选择$$\hat Q$$中最大的一个`action`。但是这种情况下$$\hat Q_t(a)$$的估计结果是相当不准确的，因为我们只重复尝试其中的一个或几个看起来短期`reward`比较高的`action`，但是却不知道那些从来没有选过的`action`是否会带来更多的收益。但是一直进行`explore`会导致我们总是进行相对随机的`action`，我们一直在探索，最终可能将$$\hat Q$$估计到十分接近$$q_*(a)$$的程度。但是这样我们很少选择最优决策方案，这就会导致总收益不高。这里面临的问题就是如何进行`explore`和`exploit`来保证累计`reward`达到最高。

UCB介绍这个问题时采用的例子是，我们每天去吃饭都需要选择餐厅。假设我们面前有两个选择，一个是我们经常去的餐厅，另一个是我们从没有去过的餐厅。对于我们每天都去的餐厅，我们对于吃什么东西是非常了解的，因此我们一般来讲会吃的比较满意；但是对于从未去过的餐厅，这里可能并不好吃，但是也有可能有更佳选择。我们一直选择其中的一个都不是最佳决策，因此需要在两个选项之间权衡。相似的，



<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/exploration_vs_exploitation.png" style="display: inline-block;" width="500"/>
</div>
正是因为两个选择在一个时间步内只能选择其中的一个，两者的关系经常被形容为`conflict`

在各种条件下，两者的取舍是取决于当前的估计$$\hat Q$$、`uncertainty`、剩余时间步数等参数的复杂相互关系的。两者之间的用于平衡取舍的算法很多，甚至有很多十分精细复杂的算法，但是绝大多数都对应用情境与先验知识做了强有力的假设，要么就不能在实际中验证，要么就不能在全面考虑问题时不可验证。当假设不成立的时候，实际效果也不明显，最优性、有界损失等难以令人满意。这里首先从最简单的开始说起。

### Action - Value Methods

### Epsilon Greedy

简单的看几个估值算法。一个`action`的真正`value`前面用$$q_*(a)​$$进行表示，代表的是收益的期望。一种最简单的方式就是在计算的每一个时间步进行如下计算
$$
\begin{equation*}
\begin{split}
Q_t(a)= & \frac{sum\:of\:rewards\:when\:a\:is\:taken}{number\:of\:times\:when\:a\:is\:taken} \\= & \frac{\sum\limits_{i=1}^{t-1}R_i \mathbb{1}(A_i=a)}{\sum\limits_{i=1}^{t-1}\mathbb{1}(A_i=a)}
\end{split}
\tag{2}
\end{equation*}
$$

其中的函数$$\mathbb{1}(\cdot)​$$表示逻辑函数，当括号内为真时返回1，否则返回0。一般的初始值$$Q_1(a)​$$均为$$0​$$，当次数无穷多时，根据大数定律，被选择了无穷多次的`action`的`value`估计值$$\hat Q​$$就会逼近$$q​$$，即真值。这种方法被称为**采样平均法**（`sample-average method`）

显然这种方法并不是最优方法，但是接下来我们就先利用这个方法进行`action`的选择。一般的为了短期最优`reward`回收，会采用一个`greedy action`，即估计值最高的`action`，如果此时同时存在若干相同取值的就随机在值最大的`action`中选取一个
$$
\begin{equation}
\begin{split}
A_t=\underset{a}{\operatorname{argmax}}Q_t(a)
\end{split}
\tag{3}
\end{equation}
$$
这种方法就被称为`greedy action selection`。这种方法利用了现有的知识去最大化`immediate reward`

当然以上操作只做了`exploit`，但是很明显这个方法是不好的，不做`exploration`的话我们可能永远也不会知道最好的那个选项是哪个。那么对上述算法的一点小小改进就是，在决策时依某一小概率$$\varepsilon$$进行选择`exploration`

这样在大部分时候（$$1-\varepsilon$$）我们都进行`exploit`，利用现有知识进行决策，而少部分时候（$$\varepsilon$$）我们进行`explore`，探索是否存在更好的解决方案。这个方法被称为$$\varepsilon-greedy$$方法，其优点在于，在时间无限长的条件下，每一个选项都有无穷多被选中的次数，也因此$$\hat Q$$能够依**大数定律**正确地收敛到$$q$$。这也保证了我们能够在有限的时间内让结果更加接近最优解。

当然这个结果也是一个渐近保证，实际应用中由于收敛速度较慢，优势也并不高。

我们来简单看一下结果，代码在我的GitHub中有，过几天加个链接。以下测试结果是在一个`10 armed testbed`上进行的，这是一个十臂赌博机（或者说是十个赌博机），每个赌博机的输出都是以某个均值正态分布的，方差为1；每个赌博机输出的均值是零均值高斯分布的，我们最终选出的结果为

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/10armedtestbed.png" style="display: inline-block;" width="500"/>
</div>

这个是我们的赌博机配置，然后就要设计程序在没有先验知识的条件下，更快更好地找出最优决策方案。我们实验是基于$$1000$$个时间步进行的，然后借助各态历经性的思想，我们进行2000次反复试验，并将结果进行平均，得到结果如下

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/10armedbanditresult.png" style="display: inline-block;" width="500"/>
</div>
显然我们可以看出以下结论

- 仅贪婪的方法，前期上升很快，但是很快就被$$\varepsilon -greedy​$$方法超越，而且最终稳定在一个较低的取值
- $$\varepsilon -greedy$$方法中，当$$\varepsilon =0.1$$时，上升速度比$$\varepsilon =0.01$$快，但是最终稳定值会小于$$\varepsilon =0.01$$的最终稳定值

如果分析一下容易发现，$$\varepsilon =0.1$$时策略有$$90\%$$的时间在执行，其他时间都在探索，并没有选择最优的方案。这样下去，最终的最大值只有理论最大值的$$91\%$$，而$$\varepsilon =0.01$$时理论最大值能够达到$$99\%$$以上。

如果赌博机输出方差更小，例如为$$0$$，那么普通的$$greedy$$决策就已经可以使用；如果方差很大，例如$$10$$，那还是$$\varepsilon-greedy$$方法更好。

以上讨论都是基于`stationary`的条件进行的，若赌博机的收益是随时间发生变化的，则`explore`就是更加必要的，此时问题更加复杂，但是强化学习问题种绝大多数情况都是非平稳的。

### Incremential Implementation

如果将每一次的`action`都记录在内存中，每次运算的时候都进行求和，那显然是没有效率的。这时随着时间的增加，算法所需要的时间与空间复杂度都会明显增加。目前为止讲的这俩都是每个时间步直接求和的暴力方法，这样效率就很低了，因此我们需要更加可靠更加有效的方式。

对于单步`action`，实际上在执行前进行估值时，是可以这样描述的
$$
\begin{equation}
\begin{split}
Q_n=\frac{R_1+R_2+...+R_{n-1}}{n-1}
\end{split}
\tag{4}
\end{equation}
$$

这个表示方法中，$$Q_n$$是某个`action`被选择$$n-1$$后的估计。这个公式就可以写作如下形式

$$
\begin{equation}
\begin{split}
Q_{n+1}=&\frac{1}{n}\sum\limits_{i=1}^{n}R_i\\=&\frac{1}{n}(R_n+\sum\limits_{i=1}^{n-1}R_i)\\=&\frac{1}{n}(R_n+(n-1)\frac{1}{n-1}\sum\limits_{i=1}^{n-1}R_i)\\=&\frac{1}{n}(R_n+(n-1)Q_n)\\=&Q_n+\frac{1}{n}(R_n-Q_n)
\end{split}
\tag{5}
\end{equation}
$$

这个结果十分重要。我们可以看出，此时我们每一步只需要保留$$n$$和$$Q_n$$，接收到新的$$R_n$$之后就可以继续计算并更新上述的两个参数，等待下一轮时间步即可。

该形式在后面将多次出现，其通用形式为

$$
\begin{equation}
\begin{split}
NewEstimation=OldEstimation+StepSize[Target-OldEstimation]
\end{split}
\tag{6}
\end{equation}
$$

其中我们称$$Target-Estimation$$为`error`，上述$$Target$$也实际上就是`reward`。这里的$$StepSize$$的实际含义是每一次参数更新时步长大小，常用$$\alpha$$来表示，一般的$$\alpha_t(a)=\frac{1}{n}$$。

> 注意这里说的$$t$$和$$n$$的含义，$$t$$就永远表示时间步，$$n$$代表的是选择指定`action`的次数。

### Nonstationary Problems

取样平均方法只在平稳问题中有效，但是假如问题并不是平稳问题，就需要考虑其他算法。这时候，简单来说，我们更应该考虑`recent reward`。由于情况总是发生改变，因此之前计算的结果就显得没有什么效果，我们更应该重视最近几次操作的经验，从而进行下一步决策。这时我们将估值的原式写作

$$
\begin{equation}
\begin{split}
Q_{n+1}=Q_n+\alpha(R_n-Q_n)

\end{split}
\tag{7}
\end{equation}
$$

这个写法就和目前比较流行的机器学习方法对应了。这里我们首先考虑步长$$\alpha$$是一个常量，此时我们拆开表达式可以得到

$$
\begin{equation}
\begin{split}
Q_{n+1}&=Q_n+\alpha(R_n-Q_n)\\&=(1-\alpha)^nQ_1+\sum_{i=1}^n\alpha(1-\alpha)^{n-i}R_i
\end{split}
\tag{8}
\end{equation}
$$

这样我们就可以看出，随着执行次数的增加，之前执行的经验会被淡化，而近期的`action`得到的`reward`权重就比较大。这个过程实际上是一个加权过程，其随时间衰减是指数衰减的，因此也被称为**指数下降加权平均法**

当我们的$$\alpha=\frac{1}{n}​$$时，表达式即上面推导的$$(5)​$$的形式，此时由前面我们推导的结论，最终估计值$$\hat Q​$$一定会依概率1收敛到$$q​$$，但实际上并不是所有的$$\{\alpha\}​$$都能保证收敛，收敛时必须满足两个条件

$$
\begin{equation}
\begin{split}
\sum_{n=1}^{\infty}\alpha_n(a)=\infty
\end{split}
\tag{9}
\end{equation}
$$

$$
\begin{equation}
\begin{split}
\sum_{n=1}^{\infty}\alpha_n^2(a)<\infty
\end{split}
\tag{10}
\end{equation}
$$

以上两个条件分别需要保证$$\alpha$$足够大，防止初始条件与随机波动因素对收敛产生影响，且保证$$\alpha$$足够小，保证算法能最终收敛。直观上是这样的结论，实际证明较为复杂。显然$$\alpha=\frac{1}{n}$$是满足条件的，而$$\alpha$$为定值的时候是不满足的，即算法最终不会收敛到固定值，而是随着`non stationary`的环境（`reward`）不断改变。

实际上对于`non stationary`条件下的问题，这正是我们所需要的。对于多数强化学习问题而言，实际也都是`non stationary`的。

当然即使是满足上述条件，收敛速度也可能是极为缓慢的，往往还需要算法执行者的调整，以便收敛到一个合适的值，实际应用中往往也不会使用这种方法。

### Optimistic Initial Value

目前介绍的方法在某种程度上都依赖于初始值（$$\hat Q_1$$）的作用，用统计学的方法说，这些方法的估计值是有偏的。对于样本平均法，一旦所有的`action`都被使用一次后，`bias`就会消失；但是如果我们采用定值$$\alpha$$，则`bias`是一个永恒的问题，只会随时间减少而不消失。

实际应用来看，`bias`问题往往不是影响，有时甚至十分有用，当然仅限于用户自定义的条件下（前面都是默认为0）。用户自定义初始值一定程度上会给我们一些先验知识，从而使决策问题能够更快解决。

人为定义`action value`是一种激励搜索的方法。假如我们全部设置为$$+5$$，则对于前期每一次`greedy`决策，我们都会在更新$$\hat Q$$的时候降低值。这样就会导致前期进行多次探索`explore`，从而收集更多的初始信息。

我们仅将这种方法看做是一个`trick`，且只是在`stationary`的条件下使用。在非稳恒条件下这个也不是一个有效方法。

某种意义上说，在初始状态上动手脚的方法并不能对非平稳的条件产生影响，因为初始条件很可能只出现一次，此后并不会继续出现。我们工作的重点也不应该放在这种问题上。

尽管如此，目前看到的和后面即将遇到的几个算法都很简单，其中的一个或若干个的组合经常能够胜任许多实际任务。



### Upper-Conﬁdence-Bound Action Selection

这个算法叫做上置信界算法。

我们的`explore`是必要的，因为估计值始终存在不确定性。前面说到的`greedy action`在一开始的时候表现较好，但是长期来看他的表现就一直是那个样子，常常会忽略许多可能更好的`non-greedy action`。虽然我们后来设置了一个小概率$$\varepsilon$$来强制进行`explore`，但是这个是真实随机，从来不会考虑对某些选项做出偏袒。但是实际上我们选择的时候应该基于每一个`non-greedy action`的潜在价值进行，一种简单的方式就是通过以下方式进行

$$
\begin{equation}
\begin{split}
A_t=\underset{a}{\operatorname{argmax}}\left[ Q_t\left(a\right) +c\sqrt{\frac{\ln t}{N_t(a)}} \right]
\end{split}
\tag{11}
\end{equation}
$$
其中$$\ln$$是自然对数，$$N_t(a)$$仍然是某个指定`action`在时间步$$t$$之前被选择的次数，因子$$c$$控制了`explore`的程度。如果 $$N_t(a)=0$$，那么`a`的决策就是贪婪方法。

上置信界`action`的主要想法是，这个平方根项代表了对行动`a`的不确定性（或方差）的度量。算式中被最大化的就是行为`a`可能的真实值的上限，其中`c`决定了置信水平。每当`a`被选择的时候，这个不确定性就会下降。直观上来看，每一个时间步中，如果我们没有选择`a`这个行为，那么分母就不会改变，但是分子就会增加，如此一来增加了不确定性度量值；如果我们选择了，那么分母会更快增加，降低了不确定性度量。这个与直观的理解是相同的。我们可以看到，分母采用的是自然对数，这个增长是逐渐减缓的，但是分母的增长永远是线性的。实际上这个算法有严格的推导过程，但是这里并不想展开描写。

如果只看测试结果的话，我们可以看出，`UCB`算法的效果显然更好一些，但是如果要将结果扩展到本书其余部分的更加普遍的强化学习问题中，这个算法的实现会更加困难。

- 非平稳问题处理上会更加困难，需要比上述解决方法中更为困难的解决方法。
- 状态空间较大时，例如使用函数近似的时候，这个算法往往是不实用的。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/UCBresult.png" style="display: inline-block;" width="500"/>
</div>

### Gradient Bandit Algorithm

目前为止本章已经讲了估计行动的`value`并利用这些值来选择`action`的方法。这通常是一种不错的方法，但是却不是唯一方法。在本节中，我们将考虑每一个`action`的数字偏好（`Numerical Preference`），我们将其表示为$$H_t(a)$$。这个偏好越大，采取行动的次数越多，但这个偏好值对于`reward`没有代表性，只有两个`action`之间的相对偏好才是选择`action`的关键——假如我们将所有的偏好都加$$1000$$，这个对于选择哪个`action`的概率其实没有什么卵影响。选择`action`的概率是由`Softmax`分布决定的，也称为`Gibbs`分布或`Boltzmann`分布
$$
\begin{equation}
\begin{split}
Pr\{A_t=a\}=\frac{e^{H_t(a)}}{\sum\limits_{b=1}^k e^{H_t(b)}}=\pi_t(a)
\end{split}
\tag{12}
\end{equation}
$$
在这里我们引入了一个$$\pi_t(a)$$符号，表示在时间步`t`采取行动的概率分布。最初所有动作的偏好值$$H_t(a)$$都是相同的，例如全部为$$0$$，使得所有动作都有相同的被选择的概率。

基于梯度上升的思路，我们自然提出了一种算法，即在每一个时间步中，当我们选择了`action`$$A_t$$并收获了`reward`$$R_t$$之后，我们对偏好值作如下更新：

$$
\begin{equation}
\begin{split}
H_{t+1}(A_t)=&H_t(A_t)+\alpha(R_t-\overline{R_t})(1-\pi_t(A_t))\\
H_{t+1}(a)=&H_t(a)-\alpha(R_t-\overline{R_t})\pi_t(a)
\end{split}
\tag{13}
\end{equation}
$$
上式对当前时间步的`action`操作，下式对该时间步没有选择的`action`做更新。$$\alpha​$$是一个尺度参数，仍然是前面描述的学习率。其中我们又定义了一个新的符号$$\overline{R}_t​$$，代表目前所有`reward`的平均值（包括时间步`t`的）。该参数的计算可以通过前面说的增量法计算（对于非平稳问题还可以采用前面说的改进方法）。这一项的意义是，作为一个参考基准线。如果当前时间的`reward`高于这个参考值，那我们认为这个选项是更有潜力的，我们在后续概率选择的时候会更加优先选择这一项，因此我们增加它的偏好值$$H_t(A_t)​$$；如果当前的`reward`比这个参考值低，我们就会做相反的操作。在做这些操作的同时，本时间步没有选择的`action`与选择的$$A_t​$$做反向更新。

我们可以尝试为赌博机做出一定的修改，例如我们将赌博机输出的概率分布改成均值为$$4$$单位方差的正态分布，实际结果没有任何变化——`Gradient Bandit Algorithm`可以适应这种情况，而不是因为均值提高就变了结果。如果我们将算式中的$$\overline{R}_t$$改成$$0$$，实际效率会低得多

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/GradientBanditResult.png" style="display: inline-block;" width="500"/>
</div>
实际上该过程是**随机梯度上升**的近似。在真正的梯度上升算法中，每个`action`的偏好值可以被增量计算为
$$
\begin{equation}
\begin{split}
H_{t+1}(a)=&H_t(a)+\alpha\frac{\partial \mathbb E[R_t]}{\partial H_t(a)}
\end{split}
\tag{14}
\end{equation}
$$
其中性能的度量是`reward`的期望
$$
\begin{equation}
\begin{split}
\mathbb E[R_t]=\sum\limits_x \pi_t(x)q_*(x)
\end{split}
\tag{15}
\end{equation}
$$
虽然我们并不知道$$q_*(x)$$的具体值，但是我们能够证明，上述的公式$$(14)$$实际上和公式$$(13)$$的更新法则是等效的，即随机梯度下降的实例。下面简单推一下
$$
\begin{equation}
\begin{split}
\frac{\partial \mathbb E[R_t]}{\partial H_t(a)}=&\frac{\partial}{\partial H_t(a)}\left[\sum\limits_x \pi_t(x)q_*(x)\right]\\=&\sum\limits_x q_*(x)\frac{\partial \pi_t(x)}{\partial H_t(a)}\\=&\sum\limits_x (q_*(x)-B_t)\frac{\partial \pi_t(x)}{\partial H_t(a)}
\end{split}
\tag{16}
\end{equation}
$$
上面的这个变换看起来是不是很狗屎？因为莫名其妙的多了一个$$B_t$$作为`baseline`项却对结果似乎没产生影响？假设我们的$$B_t$$是一个与这里求和项$$x$$没有任何关系的一项，这时注意到$$\sum\limits_x \frac{\partial \pi_t(x)}{\partial H_t(a)}=0$$，因此这里只要是一个与求和变量没关系的`baseline`就能保证等式仍然成立。这个实际上是一个假设，我们认为梯度和为$$0$$，因为随着我们每一次修正参数时概率值的变化，虽然有的增大有的减小，但是总的概率和仍然为$$1$$，概率的变化值的和也因此永远为$$0$$。然后继续做变换
$$
\begin{equation}
\begin{split}
\frac{\partial \mathbb E[R_t]}{\partial H_t(a)}=&\sum\limits_x (q_*(x)-B_t)\frac{\partial \pi_t(x)}{\partial H_t(a)}\\=&\sum\limits_x (q_*(x)-B_t)*\pi_t(x)\frac{\partial \pi_t(x)}{\partial H_t(a)}/\pi_t(x)
\end{split}
\tag{17}
\end{equation}
$$
现在这个是一个期望的形式了（因为$$\pi_t(x)$$是一个权重项），因此原式可以写作
$$
\begin{equation}
\begin{split}
\frac{\partial \mathbb E[R_t]}{\partial H_t(a)}=&\sum\limits_x (q_*(x)-B_t)*\pi_t(x)\frac{\partial \pi_t(x)}{\partial H_t(a)}/\pi_t(x)\\=&\mathbb{E}\left[ (q_*(A_t)-B_t)\frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t) \right]
\end{split}
\tag{18}
\end{equation}
$$
注意我们已经将表达式写成了期望的表达方式，进一步将随机变量$$A_t$$在期望中化简可以得到
$$
\begin{equation}
\begin{split}
\frac{\partial \mathbb E[R_t]}{\partial H_t(a)}=&\mathbb{E}\left[ (q_*(A_t)-B_t)\frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t) \right]\\=&\mathbb{E}\left[ (R_t-\overline R_t)\frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t) \right]
\end{split}
\tag{19}
\end{equation}
$$
这里我们选择`baseline`为一个特定值了，并且用$$R_t​$$代替了之前的$$q_*​$$项。毕竟这也是很显然的。然后我们带入$$\frac{\partial \pi_t(x)}{\partial H_t(a)}=\pi_t(x)(\mathbb{1}(a=x)-\pi_t(a))​$$。这个$$\mathbb{1}(\cdot )​$$我们前面已经定义过了。

> 这里插一句，这个求导实际上是对`softmax`函数的求导，其过程我懒得打公式了，大家随便一推就出来了。注意这里加了判断条件，判断$$a=x$$，这个在求导过程很重要。

那么我们就可以把这个结果带入到$$(14)$$中，得到
$$
\begin{equation}
\begin{split}
H_{t+1}(a)=&H_t(a)+\alpha(R_t-\overline R_t)(\mathbb{1}(a=x)-\pi_t(a))
\end{split}
\tag{20}
\end{equation}
$$
我们刚刚表明，`Gradient Bandit Algorithm`的偏好值更新等于其梯度，因此该算法是随机梯度上升的实例。 这确保了该算法具有稳健的收敛特性。

需要提到的是，我们这里不对所选的`action`以及`reward baseline`做任何属性的要求。例如我们可以将它们设置为$$1000$$，也可以设置为$$0$$，他们仍然是随机梯度上升算法，这些取值不会影响到参数更新的期望，但是会影响到方差，也会因此影响到收敛速度，这一点在以后会说明。将其取值为`reward`的期望值可能并不是最优选择，但是实际中很好用。

### Associative Search（Contextual Bandits）

到本章目前为止，我们讨论的都是非关联性任务，既不需要将不同的`action`与不同的`situation`联系起来的任务。在这些任务中，学习的智能体要么在平稳任务中找出单一的最优决策，要么就在非平稳任务中试图跟踪最优决策方案的变化轨迹。然而，在一个通常的强化学习任务中，我们往往不仅仅一个情况，我们的目标是学习一个决策策略：在若干情境中，最优的情境到行为的映射关系。我们这里讨论一个最简单的将非关联性任务扩展到关联性任务的方法。

假如你面前有若干不同的`k-armed bandit`任务，你每一个时间步内都需要面对其中的一个（随机）。那么因此我们的任务随着时间将随机变动到另一个任务。这个对于你来说将是一个非平稳的`k-armed bandit`问题，且随着时间步的变化，你的真值表也会随之改变。我们可以尝试使用上述的任意一种（应对非平稳问题的）方法对这种情况进行建模求解，但是实际上并不可能，除非这些真值的变化是缓慢的。现在假设，我们在每个时间步都会获得这个问题的一些辨识线索（不是真值表本身），例如我们每一个时间步操作的老虎机都会改变他的颜色，其中每一个颜色都唯一对应了一组真值表，那么我们就可以学习一个关联性任务决策策略了。当我们在不同问题上应用了正确的策略后，我们就会做得更好一些。

这是一个关联性搜索任务的例子，取名的原因是它既包含了反复试错的训练（即对最优解的`search`），又包括了`action`和与其适应的任务情况的关联`association`。这个任务目前经常被叫做`Contextual Bandits`。

### Summary

这里不想细说了。本章介绍了若干的用于`k-armed bandit`的各种常见情况的算法，其中不同算法在不同问题上有着不同的优势。我们很自然的会提问，究竟哪种方法才是最好的方法，虽然这是一个基本上不能回答的问题，但是在我们上述的统一测试平台上测试比较性能是可实现的。但是他们每个算法都有自己的参数，其性能表现往往是这种参数的函数，因此我们需要考虑参数的变化引发的性能改变。下图通过1000次平均结果对结果进行比较，其中横轴上也标注了各种参数变化，曲线颜色与参数颜色是对应的。所有的算法最终都表现出倒立`U`形的结果，其优劣由读者自己判断，不过总的来说这里看起来还是`UCB`算法比较好。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/SummaryII.png" style="display: inline-block;" width="500"/>
</div>
尽管我们介绍的算法都是比较简单的算法，但是我们可以认为这些算法都是最为先进的，有些复杂的算法，由于其复杂性和假设使得它们对我们研究的完整强化学习问题没有什么实际帮助。后面也会提出一些解决完整强化学习方法的方法，其中会部分使用本章中说到的方法。

本章方法还不能较好地让人满意地解决`explore`与`exploit`的问题。这里介绍一个已经充分研究的相关方法，是计算一个特殊的`action value`，被称为是`Gittins index`。在某些重要的特殊情况下，这个方法的计算容易处理，并且会直接指向最优解。但是这种方法需要我们对问题本身做出充分了解，因此我们通常认为这种方法是不可用的。

另外这里也不对一些后面会涉及的理论做出过分介绍，**本章到此结束**。

# Reference

[1]  [Reinforcement Learning: An Introduction](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLIntro2nd2018.pdf)

[2]  [UCB Algorithm](http://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/)