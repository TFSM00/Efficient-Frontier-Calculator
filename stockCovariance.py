from stockExpectedReturn import stdDeviation
from stockStatistics import stockReturns
import numpy as np
import pandas as pd
from tabulate import tabulate

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def stockCovariance(tickerList):
    deviations = stdDeviation(tickerList)
    returns = stockReturns(tickerList)
    table_data = {}

    for i in tickerList:
        stockCovData = []
        for j in tickerList:
            covariance = deviations[tickerList.index(i)]*deviations[tickerList.index(j)]*np.corrcoef(returns.get(i),returns.get(j))[0,1]
            stockCovData.append(covariance)

        table_data[i]=stockCovData

    covariance_table_raw = pd.DataFrame.from_dict(table_data)
    covariance_table_raw.insert(0,"Tickers", tickerList)
    covariance_table = covariance_table_raw.to_string(index=False)

    return covariance_table


def stockCovarianceTable(tickerList):
    deviations = stdDeviation(tickerList)
    returns = stockReturns(tickerList)
    table_data = {}

    for i in tickerList:
        stockCovData = []
        for j in tickerList:
            covariance = deviations[tickerList.index(i)]*deviations[tickerList.index(j)]*np.corrcoef(returns.get(i),returns.get(j))[0,1]
            stockCovData.append(covariance)

        table_data[i]=stockCovData

    covariance_table = pd.DataFrame.from_dict(table_data)
    covariance_table.insert(0,"Tickers", tickerList)
    print(tabulate(covariance_table,headers=tickerList, tablefmt="rst",showindex=False))


if __name__ == "__main__":
    stockCovarianceTable(tickers)

