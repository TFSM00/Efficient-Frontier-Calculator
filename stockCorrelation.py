from stockStatistics import stockReturns
import numpy as np
import pandas as pd
from tabulate import tabulate

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def stockCorrelation(tickerList):
    returns = stockReturns(tickerList)

    table_data = {}

    for i in tickerList:
        stockCorrData = []
        for j in tickerList:
            coef = np.corrcoef(returns.get(i),returns.get(j))[0,1]
            stockCorrData.append(coef)

        table_data[i]=stockCorrData

    correlation_table = pd.DataFrame.from_dict(table_data).set_index([pd.Index(tickerList)])
    
    return correlation_table

    
def stockCorrelationTabulate(tickerList):
    returns = stockReturns(tickerList)

    table_data = {}

    for i in tickerList:
        stockCorrData = []
        for j in tickerList:
            coef = np.corrcoef(returns.get(i),returns.get(j))[0,1]
            stockCorrData.append(coef)

        table_data[i]=stockCorrData

    correlation_table = pd.DataFrame.from_dict(table_data)
    correlation_table.insert(0,"Tickers", tickerList)
    
    print(tabulate(correlation_table,headers=tickerList, tablefmt="rst",showindex=False))

if __name__ == "__main__":
    stockCorrelationTabulate(tickers)