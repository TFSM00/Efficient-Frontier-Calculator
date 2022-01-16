from stockExpectedReturn import stdDeviation
from stockStatistics import stockReturns
import numpy as np
import pandas as pd

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def stockCovariance(tickerList):
    """
    Returns the covariance table for the tickers. The covariance is calculated for each ticker pair
    Covariance = StdDev(stock1) * StdDev(stock2) * ReturnsCorrelation(stock1,stock2)
    """

    deviations = stdDeviation(tickerList) # STD.Dev List
    returns = stockReturns(tickerList) # Returns List
    table_data = {}

    for i in tickerList:
        stockCovData = []
        for j in tickerList:
            covariance = deviations[tickerList.index(i)]*deviations[tickerList.index(j)]*np.corrcoef(returns.get(i),returns.get(j))[0,1]
            stockCovData.append(covariance)

        table_data[i]=stockCovData

    covariance_table = pd.DataFrame.from_dict(table_data).set_index([pd.Index(tickerList)])
    return covariance_table

    
if __name__ == "__main__":
    stockCovariance(tickers)


