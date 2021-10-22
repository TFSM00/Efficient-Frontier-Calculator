import yahoo_fin.stock_info as yf
import numpy as np
import statistics as stats

market = "^GSPC"

def marketStats(ticker):   
    data = yf.get_data(ticker, start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY
    
    prices = data["adjclose"].tolist()
    returns = []
    
    for i in range(0, len(prices)-1):
        returns.append(float((prices[i+1]/prices[i])-1))
    
    average = round(np.mean(returns),8)
    standardDeviation = round(stats.pstdev(returns),8)
    
    return {"Average": average, "Standard Deviation": standardDeviation}

def getMarketPrice():
    data = yf.get_data("^GSPC", start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY

    return data

def marketReturns():   
    data = yf.get_data("^GSPC", start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY
    
    prices = data["adjclose"].tolist()
    returns = []
    
    for i in range(0, len(prices)-1):
        returns.append(float((prices[i+1]/prices[i])-1))
    
    return returns

def marketReturnswithDates():
    data = yf.get_data("^GSPC", start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY
    
    del data["open"], data["close"], data["high"], data["low"], data["volume"], data["ticker"]
    
      
    prices = data["adjclose"].tolist()
    returns = [None]
    
    for i in range(0, len(prices)-1):
        returns.append(float((prices[i+1]/prices[i])-1))
        
    data["S&P500"]=returns
   
    del data["adjclose"]
    data.drop(data.head(1).index,inplace=True)
    return data



if __name__=="__main__":
    print(marketStats(market))