from stockStatistics import stockReturnsforSingle
from marketData import marketReturnswithDates
import numpy as np
import matplotlib.pyplot as plt

ticker = "AAPL"

def singleStockRollingBeta(ticker):
    """
    Returns the rolling beta for a single stock. Plots the results with 2 extra lines for 95% confidence
    """
    stockRet = stockReturnsforSingle(ticker)
    mktRet = marketReturnswithDates() 

    stockRet["S&P500"] = mktRet["S&P500"] # Adds market returns to the dataframe

    betas = [None]*36       # For precision, 36 instances of data are used so 36 empty rows are added to be able to add to the dataframe
    stddevs = [None]*36
    UpperBound95 = [None]*36
    LowerBound95 = [None]*36
    const = 2.03224450931772 # Source for this constant is unclear but it needs to be used


    for i in range(0, len(stockRet["S&P500"])-36): # As it is a rolling window, we use the past 36 dates
        tickRet = stockRet[ticker][i:i+36]
        marketRet = mktRet["S&P500"][i:i+36]
        beta, stddev = np.polyfit(marketRet,tickRet,1) # Linear regression for beta and std deviation
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
    newTable = newTable[36:] # removes the Empty Rows

    plt.plot(newTable)
    plt.title(f"Rolling Beta for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Rolling Beta")

    plt.grid() # Show grid lines
    plt.legend([f"{ticker} Beta",r"95% Lower Bound", r"95% Upper Bound"])

    plt.show()


if __name__=="__main__":
    singleStockRollingBeta(ticker)
