from weightedCovariance import weightedCovariance
from weightedCovariance import weightedCovarianceEqualWeights
from stockExpectedReturn import average
from tb3ms import TB3MS_Data
import pandas as pd
import math

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def portfolioMeasuresEqualWeights(tickerList):
    """
    Returns a dict with some statistics calculated for the whole portfolio assuming equal weights 
    Measures are: variance, std. deviation, expected return and sharpe ratio
    """
    
    wgtCov = weightedCovarianceEqualWeights(tickerList)
    portfolioVariance = 0

    for i in range(0,len(tickerList)): # Portfolio variance is the sum of all stock pair variances
        for j in range(0,len(tickerList)):
            item = wgtCov.iloc[i,j]
            portfolioVariance += float(item)

    portfolioStdev = math.sqrt(portfolioVariance)

    weights = []
        
    for i in range(0,len(tickerList)): # Adds the correct weights as a list
        weights.append(float(1/len(tickerList)))

    avg = average(tickerList)

    portfolioExpectedReturn = 0

    for i in range(0,len(tickerList)):
        portfolioExpectedReturn += avg[i]*weights[i]

    TB3MS = TB3MS_Data()

    sharpeRatio = (portfolioExpectedReturn - TB3MS.get("Last Effective Monthly Rate"))/portfolioStdev

    return {"Portfolio Variance (Equal Weights)": portfolioVariance,"Portfolio Standard Deviation (Equal Weights)":portfolioStdev, "Portfolio Expected Return (Equal Weights)": portfolioExpectedReturn, "Portfolio Sharpe Ratio (Equal Weights)": sharpeRatio}


def portfolioMeasuresWeighted(tickerList,weights):
    """
    Returns the same measures but for a specific weight for each stock as a dict
    """
    
    if len(tickerList) != len(weights):
        return print("The number of tickerList and number of weights are different")
    elif round(sum(weights),8) != round(1,8):
        return print("Weights do not equal 100%")
    else:
        wgtCov = weightedCovarianceEqualWeights(tickerList)
        portfolioVariance = 0

        for i in range(0,len(tickerList)):
            for j in range(0,len(tickerList)):
                item = wgtCov.iloc[i,j]
                portfolioVariance += float(item)

        portfolioStdev = math.sqrt(portfolioVariance)

        avg = average(tickerList)

        portfolioExpectedReturn = 0

        for i in range(0,len(tickerList)):
            portfolioExpectedReturn += avg[i]*weights[i]

        TB3MS = TB3MS_Data()
        sharpeRatio = (portfolioExpectedReturn - TB3MS.get("Last Effective Monthly Rate"))/portfolioStdev

        return {"Portfolio Variance (Selected Weights)": portfolioVariance,"Portfolio Standard Deviation (Selected Weights)":portfolioStdev, "Portfolio Expected Return (Selected Weights)": portfolioExpectedReturn, "Portfolio Sharpe Ratio (Selected Weights)": sharpeRatio}


def portfolioMeasures(tickerList,weights):
    """
    Returns same measures as a dataframe
    """

    if len(tickerList) != len(weights):
        return print("The number of tickerList and number of weights are different")
    elif round(sum(weights),8) != round(1,8):
        return print("Weights do not equal 100%")
    else:
        equalW = portfolioMeasuresEqualWeights(tickerList)
        equalW_values = list(equalW.values())

        weighted = portfolioMeasuresWeighted(tickerList,weights)
        weighted_values = list(weighted.values())

        measures = pd.DataFrame([equalW_values,weighted_values],index = ["Equal Weights Portfolio","Selected Weights Portfolio"],columns=["Variance","Standard Deviation", "Expected Return", "Sharpe Ratio"])

        return measures

    

if __name__=="__main__":
    print(portfolioMeasures(tickers,[0.5,0.05,0.05,0.1,0.1,0,0,0,0.2]))