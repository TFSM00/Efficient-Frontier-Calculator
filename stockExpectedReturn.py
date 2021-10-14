from stockStatistics import stockStatistics
import pandas as pd
from tabulate import tabulate

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def Average_StdDev_Data(tickerList):
    data = stockStatistics(tickerList)

    average = []
    stdev = []

    for key in data.keys():
        average.append(data.get(key)[0])
        stdev.append(data.get(key)[1])

    table = pd.DataFrame([average,stdev], index=["Expected Return Rate","Standard Deviation"],columns=tickerList)
    
    return table

def Average_StdDev_Table(tickerList):
    data = stockStatistics(tickerList)

    average = []
    stdev = []

    for key in data.keys():
        average.append(data.get(key)[0])
        stdev.append(data.get(key)[1])

    table = pd.DataFrame([average,stdev], index=["Expected Return Rate","Standard Deviation"],columns=tickerList)
    
    
    print(tabulate(table,headers=tickerList,tablefmt="rst"))


def stdDeviation(tickerList):
    data = stockStatistics(tickerList)
    stdev = []
    for key in data.keys():
        stdev.append(data.get(key)[1])
    
    return stdev

if __name__ == '__main__':
    Average_StdDev_Table(tickers)
