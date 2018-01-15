import matplotlib.pyplot as plt
import LoadData
import regression
from numpy import *


# m 条数据 n个属性
X,Y,attNum,trainingSampleNum=LoadData.loadDataSet('abalone.txt')  # X是 m*n  Y 是 m*1
theta=regression.standRegres(X, Y) # theta 是n*1 训练数据直接当测试数据
print('线性回归系数为:',theta)
YHat= X * theta # YHat 是 m*1
print('标准线性回归预测y值的转置为:',YHat.T)
for i in range(trainingSampleNum):
    print('真实值:',Y[i],'预测值:',YHat[i])

#绘制原始数据
fig=plt.figure()
ax=fig.add_subplot(1,1,1) #add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
ax.scatter(X[:,1].flatten().A[0].T,Y.flatten().A[0].T)
#scatter第一个参数是x坐标，第二个是y坐标，都是array类型的
#第一个参数是x坐标，取的是属性矩阵的第1列，X[:,1]的结果是m*1的矩阵，flattern把m*1的矩阵变成了 1×m的矩阵，A把这个1×m矩阵
#变成了1×m的数组，取个A[0] 就是取第一行
#print(X[:,1].flatten().A.shape)  X[:,1].flatten().A是1*200的数组


#绘制预测数据
ax.plot(X[:,1].flatten().A[0],YHat.flatten().A[0],'r-')
plt.show() #不加这句不显示图像

print('皮尔逊积矩相关系数:',corrcoef(YHat.T,Y.T))
print('RSS:',LoadData.rssError(Y,YHat))
#只有等图像关闭以后才能继续执行这句话，只有1*m的行向量才能算皮尔逊积矩相关系数，否则算不了







