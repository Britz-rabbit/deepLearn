# 懒得细看，随便过过得了
# 记住：本书认为向量的默认方向为列
import torch
# x=torch.arange(4)
# print(x)
# print(len(x))

# 搞一个矩阵
# A=torch.arange(20).reshape(5,4)
# print(A)
# print(A.T)#A的转置矩阵

# 搞另一个矩阵
# B=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# 让B和B的装置对比一下
# print(B==B.T)

# 对于一个图像，用矩阵怎么表示呢？它应该是一个三通道的矩阵，你可以认为它是一个三维矩阵，也可以认为是三个二维矩阵叠加，对于rgb的数值

# 张量运算的基本性质
# 张量的二元运算不改变形状。
# print(B*B.T)

# # 使用张量的sum()来降维。单独使用sum()会将张量的所有元素求和（沿着所有轴来降低维度），所有我们要用特定方式降维。
# print(A)
# print(A.sum())
# # 等价于下面这个：
# print(A.sum(axis=[0,1]))
# print(A.sum(axis=0))#当axis=0时，是按照矩阵的列进行合并降维的，反映到张量层面则是每行张量的第n个元素求和（一列一列），这是由于一个二维张量默认的是矩阵的列。
# print(A.sum(axis=1))

# # 也可以使用求均值的方式来给矩阵降维
# # 注意：此时我们必须重新给A赋值，否则会出现如下报错：
# # RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
# # A=torch.tensor([[1.0,2,3],[4,5,6]])
# # 或者修改A的元素类型
# A=torch.tensor(A,dtype=torch.float32)
# print(A)
# # print(A.mean())
# # print(A.mean(axis=0))
# # print(A.mean(axis=1))
# # 如果你不想进行降维，只是单纯地要求个每行/列的平均数，可以将keepdim设置为ture
# C=A.mean(axis=1,keepdim=True)
# print(C)
# # print(A/(A.mean(axis=1)))#RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
# # 再保持轴不变的情况下，就可以进行广播的除了
# print(A/C)

# 使用cumsum()来直接将张量沿着某个轴进行累加运算，而不会降低维度。累加成功的行/列会添加到原矩阵的最后一行/列
# print(A.cumsum(axis=1))

# # 接下来学习点积（dot product）的概念:相同位置元素的乘积
# x,y=torch.arange(4,dtype=torch.float32),torch.ones(4,dtype=torch.float32)
# # print(x,y)
# print(x*y)
# # 哈。。稍微有些无趣的概念

# # 了解点积的概念后开始学习矩阵积的概念
# # A表示矩阵，x表示对应行数的列向量
# A=torch.arange(12,dtype=torch.float32).reshape(3,4)
# # print(A,x)
# # 使用如下写法，进行的是广播点乘的操作
# # print(A*x)
# # 如下写法才是矩阵乘法的操作。可以看到，所谓的矩阵-向量积是将A的每个行与x的每一列（x是一个列向量）中的对应元素乘积后再求和的
# # print(torch.mv(A,x))

# # 接下来学习矩阵-矩阵积。我们就用A和A的转置矩阵来学习吧
# print(A)
# print(A.T)
# # 那么矩阵-矩阵积可以看作是矩阵-向量积的延申--只不过是列向量多了点。
# # 从上面看出：矩阵-向量积的结果是一个向量，而矩阵-矩阵积的结果则是另一个新的矩阵，但是形状有可能发生改变，具体可以回忆一下考研0线性代数课本的m*n
# # 矩阵-矩阵积使用的是torch.mm(A,B)
# print(torch.mm(A,A.T))
# # 通常我们说“矩阵乘法”的时候，指的是矩阵-矩阵积，也不是简单的A*B

# 接下来我们学习范数（norm）。线性代数中，范数是一个将向量映射到标量的函数f，是除了size外用来描述一个矩阵大小的标志。范数需要满足一下几个性质：
# 1.若对矩阵的每个元素乘以一个常数，则范数也需要改变为该常数的绝对值倍。
# f(ax)=|a|f(x)
# 2.函数中A+B得到的新矩阵进入函数运算的结果，要小于等于A和B分别进入函数运算后得到的结果的和。这一点类似三角函数。
# f(x+y)<=f(x)+f(y)
# 3.范数必须是非负的。
# f(x)>=0

# # 根据以上概念，我们可以认为矩阵元素的平方和的平方根就是一个范数，深度学习中将其称之为L₂范数。使用torch.norm(u)方法来计算
# x=torch.arange(4,dtype=torch.float16)
# print(torch.norm(x))
# # 将向量元素的绝对值之和称为L₁范数。为了得到L₁范数，我们要先对矩阵中的元素加绝对值，再对这些元素求和。
# print(torch.abs(x).sum())
# # 事实上，L₁范数和L₂范数都是更一般的Lp范数的特例。对于Lp范数，咱知道有这么个东西就行了。

# 深度学习中的最优化问题通常有一个目标，而具体问题中的这个目标通常是一个范数来表示的，或者可以理解为，呃，得分？



