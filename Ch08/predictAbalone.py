import LoadData
import LocallyWeightedLR
import regression
from numpy import *

X,Y,attNum,trainingSampleNum=LoadData.loadDataSet('abalone.txt')
#YHat01=LocallyWeightedLR.testLocallyWeightedLR(X[100:199],X[:99],Y[:99],0.1) #有许多不可逆的矩阵
#theta01=LocallyWeightedLR.LocallyWeightedLR(X[129],X[:99],Y[:99],0.1) 第129个实例的xTWX矩阵不满秩
#print "129:",X[129]
#print "129 theta:",theta01
#print "129 预测值:",X[129]*theta01
YHat1=LocallyWeightedLR.testLocallyWeightedLR(X[100:199],X[:99],Y[:99],1)
YHat10=LocallyWeightedLR.testLocallyWeightedLR(X[100:199],X[:99],Y[:99],10)
YStandR=X[100:199]*regression.standRegres(X[:99],Y[:99]) #乘在了测试数据集上
#print("k=1目标函数为:",LoadData.rssError(Y[100:199],YHat01))
print("k=1目标函数为:",LoadData.rssError(Y[100:199],YHat1))
print("k=10目标函数为:",LoadData.rssError(Y[100:199],YHat10))
print("标准线性回归目标函数为:",LoadData.rssError(Y[100:199],YStandR))