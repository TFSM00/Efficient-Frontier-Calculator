from stockStatistics import stockReturnsList
from tb3ms import TB3MS_RiskFree, TB3MS_Data
import numpy as np
from marketData import marketReturns
from marketRiskPremium import marketActualReturns
import statistics as stats
import pandas as pd
import statsmodels.api as sm



tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]


def ExPostReturnsTable_EqualWeights(tickerList):
    """
    Returns a dataframe with tickers returns, ex-post returns, ex-post actual returns, market risk premium and market returns, 
    all assuming equal portfolio weights
    """
    
    returns = stockReturnsList(tickerList)

    equalws = [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]

    mainTable = returns

    for i in range(0,len(tickerList)):
        for j in range(0,len(tickerList)):
            item = mainTable.iloc[i,j]
            new_item = float(item * equalws[i])
            mainTable.iloc[i,j]=new_item


    dflenght = len(mainTable)
    exPost = []

    for i in range(0, dflenght):
        sumProd = 0
        row = list(mainTable.iloc[i])
        for j in row:
            ind = row.index(j) 
            sumProd += float(j * equalws[ind])
        
        exPost.append(sumProd)

    mainTable["Ex-Post Returns"] = exPost

    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)

    exPostActualReturns = np.subtract(exPost, riskFree)

    mainTable["Ex-Post Actual Returns"] = exPostActualReturns

    mktRet = marketReturns()

    mainTable["Market Returns"] = mktRet

    mktAcRet = marketActualReturns()

    mainTable["Market Risk Premium"] = mktAcRet

    return mainTable

def ExPostReturnsData_EqualWeights(tickerList):
    """
    Returns a dictionary with ex-post expected return, std. deviation, sharpe ratio, alpha, beta and their respective p-values.
    Also returns market expected return, std. deviation and sharpe ratio, all assuming equal portfolio weights
    """
    
    returns = stockReturnsList(tickerList)


    equalws = [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]

    mainTable = returns

    for i in range(0,len(tickerList)):
        for j in range(0,len(tickerList)):
            item = mainTable.iloc[i,j]
            new_item = float(item * equalws[i])
            mainTable.iloc[i,j]=new_item


    dflenght = len(mainTable)
    exPost = []

    for i in range(0, dflenght):
        sumProd = 0
        row = list(mainTable.iloc[i])
        for j in row:
            ind = row.index(j) 
            sumProd += float(j * equalws[ind])
        
        exPost.append(sumProd)

    exPostExpRet = round(np.mean(exPost),8)
    exPostStDev = round(stats.stdev(exPost),8)
    lastRiskFree = TB3MS_Data().get("Last Effective Monthly Rate")
    exPostSharpeRatio = float((exPostExpRet-lastRiskFree)/exPostStDev)

    mktRet = marketReturns()

    mktExpRet = round(np.mean(mktRet),8)
    mktStdDev = round(stats.stdev(mktRet),8)
    mktSharpeRatio = float((mktExpRet-lastRiskFree)/mktStdDev)

    
    data={"Ex-Post Returns": exPost, "Market Returns":mktRet}
    df = pd.DataFrame.from_dict(data)

    Y = df["Ex-Post Returns"]
    X=df["Market Returns"]

    X = sm.add_constant(X)

    model = sm.OLS(Y,X).fit()

    alpha_beta = dict(model.params)

    exPostAlpha = alpha_beta["const"]
    exPostBeta = alpha_beta["Market Returns"]

    p_values = dict(model.pvalues)

    alphaPValue = p_values["const"]
    betaPValue = p_values["Market Returns"]

    return {"Ex-Post Expected Return": exPostExpRet, "Ex-Post Standard Deviation": exPostStDev, "Ex-Post Sharpe Ratio": exPostSharpeRatio,
    "Ex-Post Alpha": exPostAlpha,"Ex-Post Beta": exPostBeta, "Ex-Post Alpha p-value": alphaPValue, "Ex-Post Beta p-value": betaPValue, 
    "Market Expected Return": mktExpRet, "Market Standard Deviation": mktStdDev, "Market Sharpe Ratio": mktSharpeRatio}
    

def ExPostReturnsTable_SelectedWeights(tickerList, weightsList):
    """
    Returns a dataframe with tickers returns, ex-post returns, ex-post actual returns, market risk premium and market returns, 
    all assuming user selected portfolio weights
    """

    if len(tickerList) != len(weightsList): #ckecks for same lenght
        return print("The number of tickers and number of weights are different")
    elif round(sum(weightsList),8) != round(1,8): #ckecks if the weights equal 100%
        return print("Weights do not equal 100%")
    else:
    
    
        returns = stockReturnsList(tickerList)


        mainTable = returns

        for i in range(0,len(tickerList)):
            for j in range(0,len(tickerList)):
                item = mainTable.iloc[i,j]
                new_item = float(item * weightsList[i])
                mainTable.iloc[i,j]=new_item


        dflenght = len(mainTable)
        exPost = []

        for i in range(0, dflenght):
            sumProd = 0
            row = list(mainTable.iloc[i])
            for j in row:
                ind = row.index(j) 
                sumProd += float(j * weightsList[ind])
            
            exPost.append(sumProd)

        mainTable["Ex-Post Returns"] = exPost

        riskFree = TB3MS_RiskFree()
        riskFree.pop(0)

        exPostActualReturns = np.subtract(exPost, riskFree)

        mainTable["Ex-Post Actual Returns"] = exPostActualReturns

        mktRet = marketReturns()

        mainTable["Market Returns"] = mktRet

        mktAcRet = marketActualReturns()

        mainTable["Market Risk Premium"] = mktAcRet

        return mainTable

def ExPostReturnsData_SelectedWeights(tickerList, weightsList):
    """
    Returns a dictionary with ex-post expected return, std. deviation, sharpe ratio, alpha, beta and their respective p-values.
    Also returns market expected return, std. deviation and sharpe ratio, all assuming user selected portfolio weights
    """

    if len(tickerList) != len(weightsList): #ckecks for same lenght
        return print("The number of tickers and number of weights are different")
    elif round(sum(weightsList),8) != round(1,8): #ckecks if the weights equal 100%
        return print("Weights do not equal 100%")
    else:
        returns = stockReturnsList(tickerList)


        mainTable = returns

        for i in range(0,len(tickerList)):
            for j in range(0,len(tickerList)):
                item = mainTable.iloc[i,j]
                new_item = float(item * weightsList[i])
                mainTable.iloc[i,j]=new_item


        dflenght = len(mainTable)
        exPost = []

        for i in range(0, dflenght):
            sumProd = 0
            row = list(mainTable.iloc[i])
            for j in row:
                ind = row.index(j) 
                sumProd += float(j * weightsList[ind])
            
            exPost.append(sumProd)

        exPostExpRet = round(np.mean(exPost),8)
        exPostStDev = round(stats.stdev(exPost),8)
        lastRiskFree = TB3MS_Data().get("Last Effective Monthly Rate")
        exPostSharpeRatio = float((exPostExpRet-lastRiskFree)/exPostStDev)

        mktRet = marketReturns()

        mktExpRet = round(np.mean(mktRet),8)
        mktStdDev = round(stats.stdev(mktRet),8)
        mktSharpeRatio = float((mktExpRet-lastRiskFree)/mktStdDev)

        
        data={"Ex-Post Returns": exPost, "Market Returns":mktRet}
        df = pd.DataFrame.from_dict(data)

        Y = df["Ex-Post Returns"]
        X=df["Market Returns"]

        X = sm.add_constant(X)

        model = sm.OLS(Y,X).fit()

        alpha_beta = dict(model.params)

        exPostAlpha = alpha_beta["const"]
        exPostBeta = alpha_beta["Market Returns"]

        p_values = dict(model.pvalues)

        alphaPValue = p_values["const"]
        betaPValue = p_values["Market Returns"]

        return {"Ex-Post Expected Return": exPostExpRet, "Ex-Post Standard Deviation": exPostStDev, "Ex-Post Sharpe Ratio": exPostSharpeRatio,
        "Ex-Post Alpha": exPostAlpha,"Ex-Post Beta": exPostBeta, "Ex-Post Alpha p-value": alphaPValue, "Ex-Post Beta p-value": betaPValue, 
        "Market Expected Return": mktExpRet, "Market Standard Deviation": mktStdDev, "Market Sharpe Ratio": mktSharpeRatio}


if __name__=="__main__":
    #print(ExPostReturnsTable_EqualWeights(tickers))
    #print(ExPostReturnsData_EqualWeights(tickers))
    #print(ExPostReturnsTable_SelectedWeights(tickers, [0.2,0.1,0.1,0.1,0.2,-0.1,-0.2,0.3,0.3]))
    print(ExPostReturnsData_SelectedWeights(tickers, [0.2,0.1,0.1,0.1,0.2,-0.1,-0.2,0.3,0.3]))