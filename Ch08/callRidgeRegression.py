import LoadData
import RidgeRegression
import regression
import matplotlib.pyplot as plt
from numpy import *

X,Y,attNum,trainingSampleNum=LoadData.loadDataSet('abalone.txt')
xStd=std(X,0) #得到标准化前X的标准差
yMean=mean(Y,0) #得到中心化前的Y的均值 yMean是1×1的矩阵
XStand,YCentered=LoadData.standardize(X),LoadData.centered(Y)

testNum=30
wMat=RidgeRegression.ridgeTest(XStand,YCentered,testNum)
for i in range(testNum):
    theta=mat(wMat[i,:]).T
    YHat1=XStand*theta+yMean #广播
    print("参数",i,"的岭回归的Error为", LoadData.rssError(Y, YHat1))
wMat=wMat/xStd #还原到没有标准化前的系数

xOne=LoadData.addAllOneColumn(X)
thetaStd=regression.standRegres(xOne,Y)
YHat2=xOne*thetaStd
print("标准线性回归的Error为",LoadData.rssError(Y,YHat2))

# 岭迹图
lambdas = [i - 10 for i in range(testNum)]
plt.plot(lambdas, wMat)
plt.show()

