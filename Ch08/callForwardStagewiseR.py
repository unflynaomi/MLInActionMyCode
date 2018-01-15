import LoadData
import ForwardStagewiseR
import regression
import matplotlib.pyplot as plt
from numpy import *

X,Y,attNum,trainingSampleNum=LoadData.loadDataSet('abalone.txt')
xStd=std(X,0) #得到标准化前X的标准差
yMean=mean(Y,0) #得到中心化前的Y的均值
XStand,YCentered=LoadData.standardize(X),LoadData.centered(Y)

numIt=5000
allWS=ForwardStagewiseR.forwardStagewiseR(XStand,YCentered,0.005,numIt)
YHat1=XStand*(mat(allWS[numIt-1]).T)+yMean
allWS=allWS/xStd
print("前向逐步回归系数最后一次迭代系数为:",allWS[numIt-1])
print("前向逐步回归的Error为",LoadData.rssError(Y,YHat1))

xOne=LoadData.addAllOneColumn(X)
thetaStd=regression.standRegres(xOne, Y) # theta 是n*1 训练数据直接当测试数据
print('线性回归系数为:',thetaStd.T)
YHat2=xOne*thetaStd
print("标准线性回归的Error为",LoadData.rssError(Y,YHat2))

plt.plot(range(numIt),allWS)
plt.show()
