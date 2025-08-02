# LightGCN



## Introduction

舍弃了GCN的特征变换（feature transformation）和非线性激活（nonlinear activation），只保留了领域聚合（neighborhood aggregation ）。 

出发点：GCN 成为协同过滤的最新技术，但是有效性的原因没有得到充分证明。GCN 的两个设计-特征转换、非线性激活 对协作过滤的性能几乎没有贡献，且会增加培训的难度并降低推荐性能；\

贡献：

* 实验证明，GCN 中的 特征变换、非线性激活对协同过滤的有效性没有积极的影响；
* 简化了 GCN 组件；
* 通过实验对比 LightGCN 和 NGCF，阐明了 LightGCN 的合理性；



## Prelimiaries

猜想：在半监督节点分类中，每个节点都有丰富的语义特征，进行多层非线性变换有利于特征学习。然而，在协同过滤中，输入只有 user-item 的交互邻接矩阵，没有额外的语义输入，在这种情况下执行执行多个非线性变换不会有助于学习更好的特征，且带来了训练的难度.



## Model

**模型框架：**

<img src="D:\PY\JUPYTER\study\Img\LightGCN_model.png" alt="LightGCN_model" style="zoom: 80%;" />



### 公式

在 LightGCN 中，不再使用特征变换和非线性激活，而是使用对称归一化加权聚合：
$$
e_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|} \sqrt{|\mathcal{N}_i|}} \, e_i^{(k)}
$$

$$
e_i^{(k+1)} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i|} \sqrt{|\mathcal{N}_u|}} \, e_u^{(k)}
$$

最终的用户、物品嵌入（加权）：
$$
e_u = \sum_{k=0}^{K} \alpha_k \, e_u^{(k)}, \quad
e_i = \sum_{k=0}^{K} \alpha_k \, e_i^{(k)}
$$
其中 $\alpha_k = 1 / (K + 1)$

$K$ 代表传播的层数，这样的做法：

- 随着层数增加，将导致过平滑的问题，故不能简单使用最后一层的嵌入；

  > 为什么会导致平滑问题？
  >
  > 因为更新时，每个节点的表示是邻居表示的加权平均。这样，每次传播都会让**邻居之间的表示更接近**，减少差异，就像把一块高低不平的地面用砂纸磨平一样。
  >
  > 虽然对推荐来说，平滑意味着
  >
  > 1. 如果两个物品被很多相似的用户交互过，那么它们的向量会更接近
  >
  > 2. 如果两个用户喜欢的物品集合相似，它们的向量也会更接近
  > 3. 可以缓解稀疏性问题（冷启动用户/物品能通过邻居获得有用信息）
  > 4. 让相似节点的嵌入靠近，有利于用内积预测偏好
  >
  > 但过于平滑也会有风险
  >
  > 如果传播太多层，就会出现 **过平滑（over-smoothing）**：所有节点的嵌入差不多，失去区分性
  >  → **这就是 LightGCN 不直接用最后一层，而是融合多层的原因**

- 不同层捕获了不同的语义信息；　　　　

- 将不同层的嵌入加权和，起到图卷积自连接的效果；



模型预测被定义为用户和项目最终表示的内积：
$$
\hat{y}_{ui}=e_u^{\top}e_i
$$


### 矩阵形式

$$
A = 
\begin{pmatrix}
0 & R \\
R^{\top} & 0 
\end{pmatrix}
$$

其中 $R$ 是 `user2item` 交互矩阵。

事实上，我们还可以对这个矩阵做一个改进，左上角的 $0$ 矩阵可以看成是一个 $u2u$ 矩阵，而右下的 $0$ 矩阵则可以看成一个 $i2i$ 矩阵。当然，如果没有 $u2u$ 或者 $i2i$ 的信息。它们默认为 $0$。

$A$ 是一个分块矩阵，其形状是这样的：
$$
\begin{bmatrix}
users \times users & users\times items \\
items \times users & items \times items
\end{bmatrix}
$$







## 损失函数 BPR Loss

BPR Loss（Bayesian Personalized Ranking Loss）是推荐系统里非常经典的一种**pairwise 排序损失函数**，常用于像 LightGCN、NeuMF 这样的**隐式反馈**任务（只有用户行为记录，没有显式评分）。

BPR loss 定义为：

$$
L_{BPR} = -\sum_{(u,i,j) \in D} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \|\Theta\|^2
$$

其中：

- $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 sigmoid 函数  
- $\hat{y}_{ui}$ 和 $\hat{y}_{uj}$ 分别是模型对 $(u,i)$ 和 $(u,j)$ 的预测分数  
- $\lambda$ 是正则化系数  
- $D$ 是训练数据中所有三元组 $(u,i,j)$ 的集合



## 评价指标

### HR（Hit Ratio）

**作用**：衡量测试集中用户的目标物品（ground truth item）是否出现在模型推荐的 Top-K 列表中。
$$
HR@K = \frac{\text{命中次数}} {\text{总用户数}}
$$
**特点**：只关心是否命中，不关心排名位置。



### NDCG（Normalized Discounted Cumulative Gain）

**作用**：衡量目标物品在推荐列表中的位置，排名越靠前，得分越高。
$$
NDCG@K = \frac{1}{\log_2(\text{rank} + 2)}
$$
其中 `rank` 是目标物品在推荐列表中的位置（从 0 开始计）。



**特点**：既考虑命中情况，又考虑位置的影响。



## Code

我们这里的任务只是一个简单的 DEMO，还未做出推荐，只是找到模型能训练出更高质量的初始 `embedding`



### 小节

列一下需要学会的 code 技巧

* 学会稀疏矩阵 `scipy.sparse` 的常见用法/API

* 学会根据交互数据表构造矩阵（一般是稀疏矩阵）

  * `user2item`
  * `item2user`
  * `ADJ`

* 学会使用 `dgl` 的接口搭建模型（`dgl` 能够非常方便地实现**消息传递**功能）

  * `LightGCN`

    * 使用 API  `fn.copy_u()` `fn.sum()` 轻松实现 `LightGCN` 的消息传递与聚合更新

    * 使用一点小 trick 能找到对应的 user 与 item 的 embedding

    * 得到每一层的嵌入并最终取平均作为最终嵌入的写法很优美，值得学习

      ```python
      all_embs = [torch.cat([users_emb, items_emb], dim = 0)] # [ [N, D] ]
      
      for i in range(self.K):
          emb = self.layers[i](g, all_embs[-1][: self.num_users], all_embs[-1][self.num_users: ])     # [N, D]
          all_embs.append(emb)                                       
      
      # 平均， [N, D]
      all_embs = torch.stack(all_embs, dim = 0).mean(dim = 0)
      ```

      

* 理解 `BPR` 损失与根据 `BPR` 损失的需求编写符合要求的 `Dataset`

  * 能够根据 `inter` 交互**列表**与 `u2i_mat` 写出正确的采样策略函数与 `__getitem__(self, index)`



* 学会用代码实现损失函数，这里的正则化我们需要自己写

### 注意事项

* 未来可以对数据更好的封装，实现更多功能，比如实现一些**映射函数的功能**！

* 对二范数平方就是常见的 L2 正则公式，理解为什么要加正则化

* 我们会使用 `scipy.sparse` 实现一个交互矩阵与一个大矩阵。

  交互矩阵即为上面 矩阵形式 章节的 $R$；而大矩阵就是 $A$ 了。

* 需要弄清楚的是我们并没有使用异构图来进行建图区分 `user` 与 `item` 。也就是说它们的索引都是从 1 开始的。那么我们获取它们的 embedding 该怎么对应起来呢？

  首先需要搞清楚的是，我们的 `src_data` 与 `dst_data` 是通过大矩阵 $R$ 得到的。并且我们建立的 `graph` 也是以 `src_data` 和 `dst_data` 建立的。

  ```python
  edge_src, edge_dst = adj.nonzero()	# [num_users + num_items, num_users + num_items]
  
  # 建立无向图
  uv_g = dgl.graph(data = (edge_src, edges_dst),
                   idtype = torch.int32,
                   num_nodes = adj.shape[0],
                   device = device)
  ```

  也就是说，这里的 `edge_dst` 的索引默认是从 `num_users` 开始的！

  我们首先对 `user_embedding` 与 `item_embedding` 进行拼接来进行前向处理。最后得到一个拼接起来的 `embedding`，但我们不难对其进行区分

  ```python
  # embedding: [N_u + N_i, D]
  user_embedding = embedding[: user_num]	# [N_u, D]
  item_embedding = embedding[user_num: ]	# [N_i, D]
  ```

  最后我们便可以使用正常的 `user_idx` 和 `item_idx` 的索引去获取对应的嵌入了！

  

  

  
