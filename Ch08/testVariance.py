import numpy as np
import matplotlib.pyplot as plt
#验证不应该除以方差
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

# Assuming there is not intercept : y = y_1 + 2 * y_2

def ridgeTest(xArr,yArr):
    xMat = np.mat(xArr); yMat=np.mat(yArr).T
    yMean = np.mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = np.mean(xMat,0)   #calc mean then subtract it off
    xVar = np.var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts,np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:]=ws.T / xVar
    return wMat

X = np.mat(np.random.rand(50,2))
y = np.mat([1,2]) * X.T

res_orig = ridgeTest(X,y)

plt.plot(res_orig)
plt.title('Original Scaling')
plt.ylabel('Values of the (estimated) coefficient.')
plt.xlabel('Value of the penalization.')
plt.show()


X[:,1] *= 10
y = np.mat([1,2]) * X.T
res = ridgeTest(X,y)

plt.plot(res)
plt.title('x_1 has been multiplied by 10')
plt.ylabel('Values of the (estimated) coefficient.')
plt.xlabel('Value of the penalization.')
plt.show()