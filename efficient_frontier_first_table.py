from stock_statistics import stockStatistics
import pandas as pd
from tabulate import tabulate
tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def avg_stdev_table(tickerList):
    data = stockStatistics(tickers)

    average = []
    stdev = []

    for key in data.keys():
        average.append(data.get(key)[0])
        stdev.append(data.get(key)[1])

    table = pd.DataFrame([average,stdev], index=["Expected Return Rate","Standard Deviation"],columns=tickerList)
    
    return table

def stdDeviation(tickerList):
    data = stockStatistics(tickers)
    stdev = []
    for key in data.keys():
        stdev.append(data.get(key)[1])
    
    return stdev

if __name__ == '__main__':
    data = avg_stdev_table(tickers)
    print(tabulate(data,headers=tickers,tablefmt="rst"))
