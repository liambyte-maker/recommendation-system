# FM

# 1. FM出现之前存在的问题

LR的不足

LR 没有考虑特征之间的关系，经验说明，交叉特征能提升模型效果

改建： 加入二阶交叉特征

<img width="592" height="101" alt="image" src="https://github.com/user-attachments/assets/4b3d2298-8b69-470b-ba16-3b95097b8786" />

如果写成这样，会产生自己乘以自己的情况，所以要固定i,让j从i+1开始
<img width="299" height="213" alt="image" src="https://github.com/user-attachments/assets/550b5a86-d057-491d-800b-8f1bfd369fd3" />

直接引入二阶交叉特征，会出现以下缺点

参数多： 假设一共有n个特征，所有特征两两交叉，只wij一项就引入了n(n-1)/2个参数

特征稀疏难以训练，xi xj会产生大量0值，梯度会为0，不进行更新

<img width="304" height="302" alt="image" src="https://github.com/user-attachments/assets/4fb1d614-58df-4f45-9da2-e87fdb23d4e6" />


泛化能力差，各个交叉之间没有什么联系，不同组合之间不共享信息

男 × 重庆 → 爱吃辣

模型学到：

w(男×重庆) 很大

但来了一个：

女 × 重庆

模型不知道怎么办。

因为：

w(女×重庆) 没训练过。

但人类会想，重庆这个特征本身很重要

##   2. FM 算法

FM初步具备了泛化能力，对于新的特征组合也有一定的推断性质, FM需要学习的参数<有很多交叉特征的LR，在这个深度学习的时代，FM的交叉性质也没有被完全替代。

FM公式， 对于每个特征xi,不只学一个权重wi

还学一个向量， 

<img width="505" height="149" alt="image" src="https://github.com/user-attachments/assets/db6ea14e-b0cc-4156-b509-e333320ead69" />

k 是 embedding 维度

FM公式变为

<img width="769" height="115" alt="image" src="https://github.com/user-attachments/assets/10221a4d-9850-4311-b04c-67f39f458c16" />

交叉权重由向量决定

举例

假设男 有向量 V(男)

重庆 有向量 V(重庆)

w(男×重庆)=V(男)⋅V(重庆)

重庆这个城市“爱吃辣”的属性强，

它的向量会体现这个特性。

那么：

即使没有见过“女×重庆”，

只要：

V(女) 和 V(重庆) 的向量相似度高，

模型也能推断出：

女×重庆 可能也偏辣。

引入隐向量的好处

参数量从n(n-1)/2降为了kn

原先参数之间没有关联，性质可以通过隐向量简历关系






















