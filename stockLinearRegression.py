import pandas as pd
from portfolioActualReturns import portfolioActualReturns
from marketRiskPremium import marketActualReturns
import matplotlib.pyplot as plt
import numpy as np

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def stockLinearRegression(tickerList):
    actual_returns = portfolioActualReturns(tickerList)
    market_actual_returns = marketActualReturns()

    ticker = tickerList[-1]
    
    stock = actual_returns[ticker]
    market = market_actual_returns["SPY"]

    

    df = pd.DataFrame([market,stock], columns=["Market",ticker])

    plt.scatter(market,stock)
    plt.title(f"{ticker} Actual Returns vs Market")
    plt.ylabel(f"{ticker}")
    plt.xlabel("Market")

    # Trendline
    z = np.polyfit(market,stock,1)
    p = np.poly1d(z)

    plt.plot(market,p(market),"r--")
    plt.show()


if __name__=="__main__":
    stockLinearRegression(tickers)