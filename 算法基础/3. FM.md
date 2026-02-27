<img width="470" height="147" alt="image" src="https://github.com/user-attachments/assets/6195de15-07d9-4108-8f9e-5e56c1e96565" /># FM

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

原先参数之间没有关联，性质可以通过隐向量建立关系

## 3. FM的改写
<img width="470" height="147" alt="image" src="https://github.com/user-attachments/assets/6c1db510-9c7a-47fe-91a9-4081eeb607ed" />

只算i<j ， 不重复，不算自己×自己

改写成 上半角=（全部-对角线）/2

<img width="758" height="185" alt="image" src="https://github.com/user-attachments/assets/89766ecf-ea96-4dd8-bbfc-443aaa19f07b" />

内积展开

<img width="421" height="159" alt="image" src="https://github.com/user-attachments/assets/bb642d6e-69ce-4a71-87df-a455f1b9bfe4" />

代进去

<img width="913" height="205" alt="image" src="https://github.com/user-attachments/assets/61e27780-f2fb-406c-bf89-940895487f1b" />

把f提到最外面

<img width="824" height="169" alt="image" src="https://github.com/user-attachments/assets/5753c1c0-ab42-4ccc-95c5-197dcbb6656e" />

这里双重枚举变成加起来求和

<img width="700" height="208" alt="image" src="https://github.com/user-attachments/assets/3cd6335d-84cf-4195-b00f-46e3ff7ae0ee" />

最终形式

<img width="791" height="232" alt="image" src="https://github.com/user-attachments/assets/6b7991e7-ec1e-44ad-ac75-6dbefc49f544" />

计算复杂度从O(Kn^2) 改写后可减少为O（Kn)

## FM的优缺点

优点： 

减少了需要学习的参数数量，要学习的参数由n^2变成了nk,k是每个特征隐向量的长度，大大减少了学习的参数量

可以应对稀疏场景，只要xi不等于0的都能训练vi,xj不等于0的都能训练vj,相当于间接训练了wij

提升扩展性

缺点：

每个特征只有一个embedding, 每个特征一个向量，所有交互共用，不同类型特征之间交叉没有区分性

FM只能对低阶特征进行建模，对于高阶特征交叉的处理能力相对较弱


## 5. FFM

FM的隐藏问题
embedding 向量之间会相互拉扯

情况 1：ESPN + Nike 经常一起点击

梯度会推动：

变大。

<img width="314" height="124" alt="image" src="https://github.com/user-attachments/assets/cfeffb2b-1cee-4b42-9359-dc7afaa4f0fd" />


为了让内积变大，梯度会：拉近它们。S让它们更相似

情况 2：Nike + 男 经常一起点击

梯度会推动：  

<img width="238" height="96" alt="image" src="https://github.com/user-attachments/assets/b3f2da45-dac9-4353-8cc7-5aaa82361d34" /> 变大

现在：

v_ESPN 被拉向 v_Nike

v_Nike 被拉向 v_男

那 v_ESPN 会被间接拉向 v_男。

但现实中：

ESPN 和 男 可能并不相关。

## FFM和FM区别

FFM模型引入了域的概念

FM中，每个特征xi只有一个隐向量，无论与其他哪个特征交叉，都用这一个隐向量， 但是FFM给每个特征准备了多套embedding, 

假设 特征欲的数量为F， 那么每个特征就有F-1个隐向量

FFM的一大缺点就是参数爆炸

FFM

优点：  相对于FM 模型，特征刻画更加精细

缺点：

时间开销大， FFM时间复杂度为O(Kn^2)

参数过多，容易过拟合，模型太强，数据太少

FFM的时间复杂度不可以优化


















































