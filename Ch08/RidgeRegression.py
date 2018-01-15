from numpy import *
#岭回归，X Y是训练数据 返回n*1的参数矩阵 theta
def ridgeRegres(X,Y,lam=0.2):
    XTX=X.T*X #T求矩阵的转置
    ridgeXTX=XTX+lam*eye(X.shape[1])
    # print("ridgeXTX的秩为:",linalg.matrix_rank(ridgeXTX))
    theta=ridgeXTX.I*X.T*Y # I是求矩阵逆的操作
    return theta

#测试testNum次岭回归，X必须标准化，Y必须中心化，返回标准化X后的系数
def ridgeTest(X,Y,testNum=30):
    wMat = zeros((testNum, X.shape[1]))
    for i in range(testNum):
        theta = ridgeRegres(X, Y, exp(i - 10))
        wMat[i, :] = theta.T # 标准化后的系数
    return wMat


