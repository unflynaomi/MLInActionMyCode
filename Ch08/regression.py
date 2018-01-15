from numpy import *


#标准线性回归，X Y是训练数据 返回n*1的参数矩阵 theta，所有测试数据都用这个参数theta与局部加权线性回归不同
def standRegres(X,Y):
    XTX=X.T*X #T求矩阵的转置
    print("XTX的秩为:",linalg.matrix_rank(XTX))
    if linalg.det(XTX)==0: #如果矩阵行列式为0,则矩阵不可逆
        print('矩阵不可逆')
        return
    theta=XTX.I*X.T*Y # I是求矩阵逆的操作
    return theta






