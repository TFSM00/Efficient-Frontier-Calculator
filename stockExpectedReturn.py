from stockStatistics import stockStatistics
import pandas as pd

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def Average_StdDev_Data(tickerList):
    """
    Returns a dataframe with the average and std. deviation for every ticker
    """
    
    data = stockStatistics(tickerList)

    average = []
    stdev = []

    for key in data.keys():
        average.append(data.get(key)[0]) #average is first value
        stdev.append(data.get(key)[1]) #stdev is the second value 

    table = pd.DataFrame([average,stdev], index=["Expected Return Rate","Standard Deviation"],columns=tickerList)

    return table

def stdDeviation(tickerList):
    """
    Returns the standard deviations as a list
    """

    data = stockStatistics(tickerList)
    stdev = []
    for key in data.keys():
        stdev.append(data.get(key)[1])
    
    return stdev

def average(tickerList):
    """
    Returns the averages as a list
    """
    
    data = stockStatistics(tickerList)
    average = []
    for key in data.keys():
        average.append(data.get(key)[1])
    
    return average

if __name__ == '__main__':
    print(Average_StdDev_Data(tickers))
