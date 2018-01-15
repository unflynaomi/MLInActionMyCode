import matplotlib.pyplot as plt
import LoadData
import LocallyWeightedLR
from numpy import *

# m 条数据 n个属性
X,Y,attNum,trainingSampleNum=LoadData.loadDataSet('ex0.txt')  # X是 m*n  Y 是 m*1
# yHat1=X[0]*LocallyWeightedLR.LocallyWeightedLR(X[0],X,Y,0.001) 单个实例做测试
# print(yHat1)
YHat1=LocallyWeightedLR.testLocallyWeightedLR(X,X,Y,1) #m*1 YHat矩阵
print('局部加权线性回归预测y值的转置为:',YHat1.T)
print('k=1皮尔逊积矩相关系数:',corrcoef(YHat1.T,Y.T))

#绘制原始数据
fig=plt.figure()
ax=fig.add_subplot(3,1,1) #add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
ax.scatter(X[:,1].flatten().A[0].T,Y.flatten().A[0].T)

#绘制预测数据 如果不排序直接预测，无法画出正确的曲线
'''ax.plot(X[:,1].flatten().A[0],YHat1.flatten().A[0],'r-')
plt.show()'''

#书上的写法
'''strInd=X[:,1].argsort(0) #按照第2列排序 返回的是m*1的矩阵，里面存了第2列的数据排序后的索引 例如 [[161][187][162][163]]
print(strInd)
xSortThreeD=X[strInd] #因为strInd是m*1的矩阵 所以很奇怪 xSortThreeD变成了三维的m*1*2矩阵,有m个1*2的矩阵
print(xSortThreeD)
print(type(xSortThreeD))
print(xSortThreeD.shape)
xSort=xSortThreeD[:,0,:] #因为xSort是三维矩阵，切片后每个矩阵只取第一行，xSort就会变成正常的按照第二列排序的m*n的二维矩阵
print(xSort)
print(YHat[strInd]) # YHat是m*1的矩阵 strInd是m*1的矩阵 YHat[strInd]是三维的m*1*1矩阵
ax.plot(xSort[:,1].flatten().A[0],YHat[strInd].flatten().A[0],'r-')
plt.show() #不加这句不显示图像'''

#绘制预测数据
index= X.A[:, 1].argsort() #array取第2列以后是一维的，argsort以后也就是个一维数组
XSort=X[index] #XSort就是按照第2列排序后的m*n二维数组
#print(XSort)
YHat1Sort=YHat1[index] #YHat1Sort就是排完序后的每个实例对应的预测值
#print(YHat1Sort)
ax.plot(XSort[:,1].flatten().A[0],YHat1Sort.flatten().A[0],'r-')

ax=fig.add_subplot(3,1,2) #add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
ax.scatter(X[:,1].flatten().A[0].T,Y.flatten().A[0].T)
YHat2=LocallyWeightedLR.testLocallyWeightedLR(X,X,Y,0.01) #m*1 YHat矩阵
print('k=0.01皮尔逊积矩相关系数:',corrcoef(YHat2.T,Y.T))
YHat2Sort=YHat2[index]
ax.plot(XSort[:,1].flatten().A[0],YHat2Sort.flatten().A[0],'r-')

ax=fig.add_subplot(3,1,3) #add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
ax.scatter(X[:,1].flatten().A[0].T,Y.flatten().A[0].T)
YHat3=LocallyWeightedLR.testLocallyWeightedLR(X,X,Y,0.003) #m*1 YHat矩阵
print('k=0.003皮尔逊积矩相关系数:',corrcoef(YHat3.T,Y.T))
YHat3Sort=YHat3[index]
ax.plot(XSort[:,1].flatten().A[0],YHat3Sort.flatten().A[0],'r-')

plt.show() #不加这句不显示图像