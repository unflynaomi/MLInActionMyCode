from numpy import *

# m 条数据 n个属性
#读数据，返回属性取值矩阵m*n 的X 和类取值矩阵 m*1 的Y
def loadDataSet(fileName):
    attList = [] #除类标签外，属性的取值列表，一行放在一个列表里，例如[[1.000000,0.067732],[1.000000,0.427810]]
    labelList = [] #真实的y的列表 : [3.176513, 3.816464, 4.550095]
    trainingSampleNum=0
    with open(fileName, 'r') as fr1:
        attNum=len(fr1.readline().split('\t'))-1 #获得属性个数
    with open(fileName,'r') as fr2:
        lines=fr2.readlines()
    for line in lines:
        numArray=list(map(float,line.split('\t')))
        attList.append(numArray[0:attNum])
        labelList.append(numArray[-1])
        trainingSampleNum=trainingSampleNum+1
    X = mat(attList)  # X是 m*n
    Y = mat(labelList).T  # Y 是 m*1
    print('属性个数:', attNum)
    print('训练实例个数:', trainingSampleNum)
    print('属性矩阵的列表:', attList)
    print('类标签列表:', labelList)
    return X,Y,attNum,trainingSampleNum


#计算模型的目标函数 Y YHat 都是m*1的矩阵
def rssError(Y,YHat):
    return ((Y.flatten().A[0]-YHat.flatten().A[0])**2).sum()

def standardize(M):
    Mmean=mean(M,0)
    Mstd_deviation=std(M,0)
    M=(M-Mmean)/Mstd_deviation
    return M

def centered(M):
    Mmean = mean(M, 0)
    M = (M - Mmean)
    return M

#在X左边加上全1列
def addAllOneColumn(X):
    X1=mat(ones((X.shape[0],X.shape[1]+1)))
    X1[:,1:X.shape[1]+1]=X
    return X1




