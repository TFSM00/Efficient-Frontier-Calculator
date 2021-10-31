import pandas as pd
import yahoo_fin.stock_info as yf
import numpy as np
import statistics as stats
import scipy.stats as st

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA","SPY"]


def stockStatistics(tickerList): 
    """
    This function gets a few statistics related to a stocks price or returns in a given period
    It outputs statistics as a dictionary
    """

    mainData={} #serves as storage for data
    
    for ticker in tickerList:
        
        data = yf.get_data(ticker, start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY
        
        prices = data["adjclose"].tolist() #only prices
        returns = [] #returns list

        for i in range(0, len(prices)-1): # -1 because it references a date and the next one so last date is impossible to calculate
            returns.append(float((prices[i+1]/prices[i])-1)) #add return to list

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
    """
    Returns statistics as pandas dataframe
    """
    
    mainData = stockStatistics(tickerList)
    statistics = ["Average","Standard Deviation","Kurtosis","Sample","Actual Kurtosis","Skewness","Jarque-Bera Test","Jarque-Bera p-value"]
    
    stockStatsTable = pd.DataFrame.from_dict(mainData).set_index([pd.Index(statistics)])
    #get table from dict and set index as statistics not the tickers as they are columns
    
    return stockStatsTable


def stockReturns(tickerList):
    """
    Returns a dictionary with the tickers and their respective returns as a list
    """

    returnsData = {}
    for ticker in tickerList:
        
        data = yf.get_data(ticker, start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY
        
        prices = data["adjclose"].tolist()
        returns = []

        for i in range(0, len(prices)-1): # -1 because it references a date and the next one so last date is impossible to calculate
            returns.append(float((prices[i+1]/prices[i])-1))
    
        returnsData[ticker]=returns
    
    return returnsData

def stockReturnsList(tickerList):
    """
    Returns a dataframe with all the stocks returns
    """

    data = yf.get_data(tickerList[0], start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY
    
    del data["open"], data["close"], data["high"], data["low"], data["volume"], data["ticker"] # deletes all but one column 
    
    for ticker in tickerList:
        datax = yf.get_data(ticker, start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY
        prices = datax["adjclose"].tolist()
        returns = [None] #adds None as first element to easily integrate the returns into the dataframe. weird solution but works
    
        for i in range(0, len(prices)-1):
            returns.append(float((prices[i+1]/prices[i])-1))
        
        data[ticker]=returns
   
    del data["adjclose"] #deletes extra column
    data.drop(data.head(1).index,inplace=True) #removes None line
    return data

def stockReturnsforSingle(ticker):
    """
    Returns dataframe with the returns for only 1 ticker 
    """
    
    data = yf.get_data(ticker, start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY
    
    del data["open"], data["close"], data["high"], data["low"], data["volume"], data["ticker"]
    
    prices = data["adjclose"].tolist()
    returns = [None] #adds None as first element to easily integrate the returns into the dataframe. weird solution but works
    
    for i in range(0, len(prices)-1):
        returns.append(float((prices[i+1]/prices[i])-1))
        
    data[ticker]=returns
   
    del data["adjclose"]
    data.drop(data.head(1).index,inplace=True)
    return data

if __name__=="__main__":
    print(stockStatisticsTable(tickers))
