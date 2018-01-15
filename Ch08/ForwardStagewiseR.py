from numpy import *
import LoadData

def forwardStagewiseR(X,Y,eps=0.01,numIt=100):
    n=X.shape[1]
    currentWS=zeros((n,1))#currentWS目前迭代的系数，n*1的初始参数矩阵全部初始化成0
    wsChoose=zeros((n,1))
    allWS=zeros((numIt,n))
    for i in range(numIt):
        set_printoptions(suppress=True)
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTmp=currentWS.copy() #不能直接在currentWS上修改，也不能wsTmp=currentWS 然后修改
                wsTmp[j,:]+=sign*eps
                YHat=X*wsTmp
                rss=LoadData.rssError(Y, YHat)
                if rss<lowestError:
                    lowestError=rss
                    wsChoose=wsTmp #wsTmp和wsChoose都指向了同一个对象，后来wsTmp离开了，但这不会改变wsChoose的指向
        currentWS=wsChoose
        allWS[i]=currentWS.T
        #print('第',i,'次迭代后系数为:',allWS[i])
    return allWS
