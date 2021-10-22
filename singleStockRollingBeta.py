from stockStatistics import stockReturnsforSingle
from marketData import marketReturnswithDates
import numpy as np
import matplotlib.pyplot as plt

ticker = "AAPL"

def singleStockRollingBeta(ticker):
    stockRet = stockReturnsforSingle(ticker)
    mktRet = marketReturnswithDates()

    stockRet["S&P500"] = mktRet["S&P500"]

    betas = [None]*30
    stddevs = [None]*30
    UpperBound95 = [None]*30
    LowerBound95 = [None]*30
    const = 2.03224450931772


    for i in range(0, len(stockRet["S&P500"])-30):
        tickRet = stockRet[ticker][i:i+30]
        marketRet = mktRet["S&P500"][i:i+30]
        beta, stddev = np.polyfit(marketRet,tickRet,1)
        betas.append(beta)
        stddevs.append(stddev)
        UpperBound95.append(float(beta + (const*stddev)))
        LowerBound95.append(float(beta - (const*stddev)))


    stockRet[f"{ticker} Beta"]=betas
    stockRet["Standard Deviation"]=stddevs
    stockRet[r"95% Lower Bound"]=LowerBound95
    stockRet[r"95% Upper Bound"]=UpperBound95

    newTable = stockRet

    del newTable[ticker]
    del newTable["S&P500"]
    del newTable["Standard Deviation"]
    newTable = newTable[30:]

    plt.plot(newTable)
    plt.title(f"Rolling Beta for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Rolling Beta")

    plt.grid()

    plt.show()



