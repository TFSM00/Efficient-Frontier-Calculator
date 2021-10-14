import pandas as pd
import yahoo_fin.stock_info as yf
import numpy as np
import statistics as stats
import scipy.stats as st
from tabulate import tabulate


tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA","SPY"]


def stockStatistics(tickerList):
    mainData={}
    
    for ticker in tickerList:
        
        data = yf.get_data(ticker, start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY
        
        prices = data["adjclose"].tolist()
        returns = []

        for i in range(0, len(prices)-1):
            returns.append(float((prices[i+1]/prices[i])-1))

        average = round(np.mean(returns),8)
        standardDeviation = round(stats.pstdev(returns),8)
        kurtosis = round(st.kurtosis(returns),8)
        sample = len(returns)
        actualKurtosis = round(float((((kurtosis*(sample-2)*(sample-3)/(sample-1)-6)/(sample+1)+3))),8)
        skewness = round(st.skew(returns, bias=True),8)
        jarque_bera_test, jarque_bera_p_value = st.jarque_bera(returns)
        
        jarque_bera_test = round(jarque_bera_test,8)
        jarque_bera_p_value = round(jarque_bera_p_value,8)

        statsList=[average,standardDeviation,kurtosis,sample,actualKurtosis,skewness,jarque_bera_test,jarque_bera_p_value]
        
        
        mainData[ticker]=statsList
        
    return mainData

def stockStatisticsTable(tickerList):
    mainData = stockStatistics(tickerList)
    statistics = ["Average","Standard Deviation","Kurtosis","Sample","Actual Kurtosis","Skewness","Jarque-Bera Test","Jarque-Bera p-value"]
    

    stockStats = pd.DataFrame.from_dict(mainData)
    stockStats.insert(0,"Statistics", statistics)
    
    print(tabulate(stockStats, headers = tickerList, tablefmt="rst", showindex=False))

def stockReturns(tickerList):
    returnsData = {}
    for ticker in tickerList:
        
        data = yf.get_data(ticker, start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY
        
        prices = data["adjclose"].tolist()
        returns = []

        for i in range(0, len(prices)-1):
            returns.append(float((prices[i+1]/prices[i])-1))
    
        returnsData[ticker]=returns
    
    return returnsData


if __name__=="__main__":
    stockStatisticsTable(tickers)
