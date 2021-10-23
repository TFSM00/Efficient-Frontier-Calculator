import pandas as pd
from portfolioActualReturns import singleStockActualReturns
from marketRiskPremium import marketActualReturns
import matplotlib.pyplot as plt
import numpy as np


def stockLinearRegression(ticker):
    stock = singleStockActualReturns(ticker)
    market = marketActualReturns()


    df = pd.DataFrame({"Market": market, ticker: stock})

    plt.scatter(market,stock)
    plt.title(f"{ticker} Actual Returns vs Market")
    plt.ylabel(f"{ticker}")
    plt.xlabel("Market")

    # Trendline
    z = np.polyfit(market,stock,1)
    p = np.poly1d(z)

    plt.plot(market,p(market),"r--")
    plt.axhline(linewidth=1, color="black")
    plt.axvline(linewidth=1, color="black")
    plt.show()


if __name__=="__main__":
    stockLinearRegression("MSFT")