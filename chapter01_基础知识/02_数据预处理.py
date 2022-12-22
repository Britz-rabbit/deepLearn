# 深度学习需要张量，但是张量不是从书上摘得，而是利用pandas进行处理

# 首先利用os来预先创建的csv文件进行一下基本处理
import os
# 在指定目录创建一个文件
os.makedirs(os.path.join('./', 'data'), exist_ok=True)
data_file = os.path.join('./', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 引入pandas来读取数据
import pandas as pd
data = pd.read_csv(data_file)
# print(data)   
# 可以看到，打印出来的数据的第一行第一列是NaN，我们利用插值法来处理一下
# 通过iloc方法来将数据分为输入和输出。那么我们将前两列（numrooms和alley）作为输入，第三列（price）作为输出
inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
# 将输入中缺失的数值用平均值代替
inputs=inputs.fillna(inputs.mean())
print(type(inputs))
# 注意：对于allay列，它只接受pave和nan两种类型的数据。并且pandas将进一步处理：先分成pava和nan两列，再对里面的值用0（有），1（无）代替
# print(data)
inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)

# 基本的引入和说明已经进行了，接下来让我们开始将这个表转换成张量，我们用pytorch来进行
import torch 
x,y=torch.tensor(inputs.values),torch.tensor(outputs.values)
print(x)
