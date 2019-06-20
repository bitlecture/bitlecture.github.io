# BIT Lecture

Under construction.

Any suggestions mail to [our public email](mailto:bitlecturepublic@163.com)



## 编辑注意事项

对于有`push`权限的各位编辑，以`Windows`为例，推荐采用`GitHub Desktop`对`repo`进行`clone`，然后使用自己喜欢的编辑器进行修改。推荐`Markdown`编辑器为`Typora`，推荐代码编辑器为`VS Code`。

由于大多数人不具备编辑基础，因此不推荐修改已有代码，而是建议仅对`.\_posts\`下的投稿进行修改。

## 投稿编辑细则

### 命名

投稿文件位置为`.\_posts\`，所有投稿文件按照规则命名

```
YYYY-MM-DD-Title.md
```

其中标题`Title`的空格也全部使用`-`填充。扩展名也可以是`markdown`。

---

### 头部标识

投稿文件需要指定的头部标识，典型的头部标识可以写作如下形式

```yaml
---
title:  "title"
excerpt: "excerpt"
date:   2019-6-20 15:31:47 +0000
categories: Notes
comments: true
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
author: Shicong Liu
---
```

以上是`yaml`表示，标题和副标题可以填写与实际相符的内容，时间正确填写，后面的`catagories`和`comments`还没有开放，可以先不填（不写这一项）。其中`catagories`可以提供一个分类方式，而`comment`可以使用第三方平台进行评论。

`toc`是侧边导航栏，默认关闭，在这里手动开启，可选图标参考官方文档即可。如果希望将内容扩展到右侧的区域（这时导航栏将嵌入到主体中），则选择添加以下条目。

```yaml
classes:  wide
```



关于作者，需要在`\_data\authors.yml`中添加作者信息。每一位投稿人都可以根据个人信息进行填写，并且如上方式添加`author`即可在页面左侧看到作者的自定义信息。注意作者的头像信息需要放在`\assets\images\`下，并且在信息中对应。这里推荐采用正方形或近似正方形，主要内容集中在中心圆形内的头像，否则可能显示不全。



---

### 内容

以上是关于文件头部的规则，关于内容，规则如下

1. 在文件开始处不需要使用一级标题再叙述一遍`title`，因为默认会将`title`设置为一级标题显示出来。
2. 

投稿中的一些新特性方法会在这里逐渐更新，希望投稿或编辑之前先查看近期的更新情况。



