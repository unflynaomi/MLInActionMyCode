from numpy import *
from bs4 import BeautifulSoup
import regression
import RidgeRegression
import LoadData
import random


def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
    函数说明:从页面读取数据，生成retX和retY列表
    Parameters:
        retX - 数据X
        retY - 数据Y
        inFile - HTML文件
        yr - 年份
        numPce - 乐高部件数目
        origPrc - 原价
    Returns:
        无
    Website:
        http://www.cuijiahua.com/
    Modify:
        2017-12-03
    """
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)

    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)  # 这句话不是多余的，因为过一会要判断(len(currentRow) != 0)
    # print(currentRow)

    while (len(currentRow) != 0):
        currentRow = soup.find_all('table',
                                   r="%d" % i)  # table标签第一行里写了r="1" 第二行里写了r="2" 寻找所有的 [<table class="li" r="1">
        title = currentRow[0].find_all('a')[
            1].text  # currentRow[0].find_all('a')[1] 是<a class="" herf="">Lego Technic 8288 Crawler crane</a> 取出文字域来就是Lego Technic 8288 Crawler crane
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0

        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')  # 寻找[<span class="sold">Sold</span>]
        if len(soldUnicde) != 0:
            #print("商品 #%d 没有出售" % i)  # i就是商品在网页上的行业
        #else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[
                4]  # 形如类似<td class="prc bidsold g-b">$399.00</td> 或者 <td class="prc"><div class="bidsold g-b">$399.95</div><span class="tfsp">Free shipping</span></td>
            priceStr = soldPrice.text  # 诸如$399.00 或者 $399.95Free shipping
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')  # 去掉千分位
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')  # 还包邮哩
            sellingPrice = float(priceStr)

            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                #print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))  # 年份，拼图数，是否是新的，原价，售价
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)  # 这句话不是多余的，因为过一会要判断(len(currentRow) != 0)


def setDataCollect(retX, retY):
    """
    函数说明:依次读取六种乐高套装的数据，并生成数据矩阵
    Parameters:
        无
    Returns:
        无
    Website:
        http://www.cuijiahua.com/
    Modify:
        2017-12-03
    """
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)  # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)  # 2009年的乐高10196,部件数目3263,原价249.99


# 岭回归交叉验证
def crossValidationRidgeRegression(X, Y, numOfFold=10):
    m = X.shape[0]
    index = list(range(m))
    testNum = 30
    err=zeros((numOfFold,testNum)) #每个fold测试30个超参数
    for i in range(numOfFold):
    # 每一次交叉验证实验
        random.shuffle(index)#shuffle不是很可靠，肯定有重复作为训练集，重复做测试集的
        # 可以使用j%numOfFold==i来划分训练集与测试集，和petal里一样
        trainX = []
        testX = []
        trainY = []
        testY = []
        for j in range(m):
            if j<m*0.9:#训练集
                trainX.append(list(X.A[index[j]])) #先把矩阵X转化成数组，取一行以后转化成了一维数组，然后才能转成list
                trainY.append(list(Y.A[index[j]]))
            else:
                testX.append(list(X.A[index[j]]))
                testY.append(list(Y.A[index[j]]))
        trainX=mat(trainX)
        trainY=mat(trainY)
        testX=mat(testX)
        testY=mat(testY)
        trainXStand,trainYCentered=LoadData.standardize(trainX),LoadData.centered(trainY)
        xTrainMean=mean(trainX,0)
        xTrainStd=std(trainX,0)
        yTrainMean = mean(trainY, 0)  # 得到中心化前的Y的均值 yTrainMean是1×1的矩阵
        testXStand=(testX-xTrainMean)/xTrainStd #用train的均值和标准差标准化
        wMat=RidgeRegression.ridgeTest(trainXStand,trainYCentered,testNum) #wMat是每一个超参数预测的模型
        for k in range(testNum):
            theta = mat(wMat[k, :]).T #必须要转化为矩阵，才能转置，否则是1维的，转置就是它本身
            testYHat = testXStand * theta + yTrainMean #broadcast
            err[i,k]=LoadData.rssError(testY, testYHat)
    meanErr=mean(err,0)
    bestLam=argmin(meanErr)
    print("通过十折交叉验证选出最好的岭回归超参数lambda是:",exp(bestLam - 10))

    #开始在所有数据上训练模型
    xStd = std(X,0)  # 得到标准化前X的标准差
    xMean=mean(X,0)
    yMean=mean(Y)
    XStand, YCentered = LoadData.standardize(X), LoadData.centered(Y)
    theta=RidgeRegression.ridgeRegres(XStand, YCentered,exp(bestLam - 10))
    YHat=XStand*theta+yMean
    unStandCoff=theta.T/xStd; #xStd是n*1的，theta是1*n的，所以需要转置，否则结果一塌糊涂
    print("岭回归的系数(已经缩放至原尺寸):",unStandCoff)
    print("岭回归的截距(已经缩放至原尺寸):",-1*sum(multiply(unStandCoff,xMean))+yMean)
    print('岭回归RSS:', LoadData.rssError(Y, YHat))




if __name__ == '__main__':
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    lgX = mat(lgX)
    lgY = mat(lgY).T
    m = lgX.shape[0]
    n = lgX.shape[1]
    lgX1 = LoadData.addAllOneColumn(lgX)
    print('属性个数:', n)
    print('训练实例个数:', m)
    set_printoptions(suppress=True)
    print('属性矩阵:', lgX)
    print('类标签矩阵矩阵:', lgY.T)

    theta = regression.standRegres(lgX1, lgY)  # theta 是n*1 训练数据直接当测试数据
    print('线性回归系数为:', theta)
    YHat = lgX1 * theta  # YHat 是 m*1
    # for i in range(m):
    #     print('真实值:', lgY[i], '预测值:', YHat[i])
    print('标准线性回归RSS:', LoadData.rssError(lgY, YHat))

    crossValidationRidgeRegression(lgX, lgY)
