from stockStatistics import stockReturns
import numpy as np
import pandas as pd
from tabulate import tabulate

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def stockCorrelation(tickerList):
    """
    Returns a correlation dataframe between all tickers
    """

    returns = stockReturns(tickerList)

    table_data = {}

    for i in tickerList:
        stockCorrData = []
        for j in tickerList:
            coef = np.corrcoef(returns.get(i),returns.get(j))[0,1] # Correlation
            stockCorrData.append(coef)

        table_data[i]=stockCorrData

    correlation_table = pd.DataFrame.from_dict(table_data).set_index([pd.Index(tickerList)])
    
    return correlation_table


if __name__ == "__main__":
    stockCorrelation(tickers)