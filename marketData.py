import yahoo_fin.stock_info as yf
import numpy as np
import statistics as stats

market = "SPY"

def marketStats(ticker):   
    data = yf.get_data(ticker, start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY
    
    prices = data["adjclose"].tolist()
    returns = []
    
    for i in range(0, len(prices)-1):
        returns.append(float((prices[i+1]/prices[i])-1))
    
    average = round(np.mean(returns),8)
    standardDeviation = round(stats.pstdev(returns),8)
    
    return {"Average": average, "Standard Deviation": standardDeviation}


if __name__=="__main__":
    print(marketStats(market))