import pandas as pd
import yahoo_fin.stock_info as yf
import numpy as np
import statistics as stats
import scipy.stats as st


tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA","SPY"]

mainData =  {}

def stockStatistics(tickersList):
    mainData={}
    
    for ticker in tickersList:
        
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

def stockStatisticsTable(tickersList):
    mainData = stockStatistics(tickersList)
    statistics = ["Average","Standard Deviation","Kurtosis","Sample","Actual Kurtosis","Skewness","Jarque-Bera Test","Jarque-Bera p-value"]
    

    stockStats_raw = pd.DataFrame.from_dict(mainData)
    stockStats_raw.insert(0,"Statistics", statistics)
    stockStats = stockStats_raw.to_string(index=False)
    return stockStats





if "__name__"=="__main__":
    print(stockStatistics(tickers))
