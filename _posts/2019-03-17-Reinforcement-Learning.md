---
layout: post
comments: true
title:  "Reinforcement Learning: Finite Markov Decision Process"
excerpt: "-"
date:   2019-03-17 14:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# Reinforcement Learning

`2019-3-17 12:03:10`

我发现这本书无论从哪个地方开始看都能看个八成懂，还是吹一下这外国的作者好了。

## 第三章 - Finite Markov Decision Process

本章主要介绍的是**有限马尔科夫决策过程**，后面可能会写作`finite MDP`。关于这个决策过程的问题，我们在整本书剩余部分都将致力于解决。这个问题涉及了评价反馈`evaluative feedback`，就像前面介绍的`bandit`问题一样，但是也包括了一些联合性问题，例如在不同的`situation`下选择不同的行为`action`。

马尔科夫决策过程是一个顺序决策的经典形式，其中每个`action`不仅影响`immediate reward`，还会影响到后续的情境`situation`，或者我们这里说，`state`，即状态，以及未来的`reward`。因此马尔科夫决策过程问题中涉及了延迟奖励问题以及我们需要对即时奖励与延迟奖励的权衡问题。在上一章中，我们估计每个`action`的真值$$q_*(a)​$$，而在马尔科夫决策过程中我们就需要估计每个状态`state`下的`action`的值$$q_*(s,a)​$$，或者我们可以对给定的若干可用`action`估计每一个`state`的值$$v_*(s)​$$。这些依赖于`state`的量对于准确地为个体行动进行长期的分配十分重要。

### 3.1 The Agent-Environment Interface

`MDP`旨在质结构件从交互中学习以实现目标的问题。学习者和决策者我们称之为`agent`，与`agent`交互的（即`agent`之外的部分）我们称之为环境`environment`。这些会持续交互，`agent`选择`action`，而`env`响应`action`并向`agent`呈现新的情况。此外环境还会产生`reward`，这就是我们`agent`通过选择不同`action`寻求最大化的量。

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/agentenv.png" style="display: inline-block;" width="500"/>
</div>

更具体地说，`agent`与`env`在一系列离散时间步中进行交互，在每个时间步`t`，`agent`会接受环境状态的某些表示形式，例如$$S_t\in \mathfrak{S}$$，并基于这个状态`state`我们会选择一个`action`。在下一个时间步中，`agent`会收到一个`reward`，其中一部分受到上一步`action`的影响，一部分受到之前所有`action`的影响，并接收到一个不同于上一步的状态$$S_{t+1}$$。这个过程因此产生一个如下的序列

$$S_0,A_0,R_1;S_1,A_1,R_2...$$

在我们所谓的有限`MDP`中，上述的三个量都存在一个有限的集合，即状态空间、动作空间以及奖励空间都是元素有限的。如果我们将$$S_t,R_t$$这种量看作是整个过程中的一个随机变量，这两个随机变量都有完好定义的离散概率分布，仅取决于前面的状态`state`和动作`action`。也就是说，对于这两个随机变量的取值，在时间步`t`时取到指定值的概率分布是可以写成如下形式的
$$
\begin{equation}
\begin{split}
p(s',r\mid s,a)=Pr\{S_t=s',R_t=r\mid S_{t-1}=s,A_{t-1}=a\}
\end{split}
\tag{1}
\end{equation}
$$



# Reference

[1]  [Reinforcement Learning: An Introduction](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLIntro2nd2018.pdf)

