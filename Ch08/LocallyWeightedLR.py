from numpy import *


#局部加权线性回归,为一个测试样本和训练数据X和Y 返回一个n*1的参数矩阵 theta
def LocallyWeightedLR(testPoint,X,Y,k=1.0):
    m=X.shape[0]
    attNum=X.shape[1]
    W=mat(eye(m)) #m*m的单位矩阵
    for i in range(m):
        diff=testPoint-X[i,:]
        W[i,i]=exp(diff*diff.T/(-2*k*k))
    partOne=X.T*W*X

    #theta=linalg.solve(partOne,X.T*W*Y)
    # print(partOne)
    # print "逆:",partOne.I
    # print(linalg.det(partOne))
    # print(linalg.matrix_rank(partOne))
    # if linalg.det(partOne)==0: #如果矩阵行列式为0,则矩阵不可逆
    #     print('矩阵不可逆',linalg.det(partOne))
    #     return

    if linalg.matrix_rank(partOne)==attNum:  #如果矩阵不满秩,则矩阵不可逆，比行列式计算要准
        theta=partOne.I*X.T*W*Y
    else:
        theta=mat([])
    return theta

#输入测试数据和训练数据X和Y 返回为测试数据矩阵预测的m*1 Y矩阵
def testLocallyWeightedLR(testData,X,Y,k=1.0):
    m = testData.shape[0]
    YHat=mat(empty((m,1)))
    for i in range(m):
        #print("第",i,"个:",end=" ")
        theta=LocallyWeightedLR(testData[i],X,Y,k)
        if theta.size!=0 :
            YHat[i,0]=testData[i]*theta
            #print(YHat[i,0])
        # else:
        #     print("矩阵不可逆")
    return YHat


