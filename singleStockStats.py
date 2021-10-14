import pandas as pd
import yahoo_fin.stock_info as yf
import numpy as np
import statistics as stats
import scipy.stats as st

ticker = "SPY"

def getStockStats(ticker):

    data = yf.get_data(ticker, start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY

    prices = data["adjclose"].tolist()

    returns = []

    for i in range(0, len(prices)-1):
        returns.append(float((prices[i+1]/prices[i])-1))

    average = np.mean(returns)
    standardDeviation = stats.pstdev(returns)
    kurtosis = st.kurtosis(returns)
    sample = len(returns)
    actualKurtosis = float((((kurtosis*(sample-2)*(sample-3)/(sample-1)-6)/(sample+1)+3)))
    skewness = st.skew(returns, bias=True)
    jarque_bera_test, jarque_bera_p_value = st.jarque_bera(returns)

    return [average, standardDeviation, kurtosis, sample, actualKurtosis, skewness, jarque_bera_test, jarque_bera_p_value]


if __name__ == "__main__": #if script runs as program not import
    stats = getStockStats(ticker)
    print("General Statistics on ticker: " + ticker + "\n")
    print("Average Returns: " + str(stats[0]))
    print("Standard Deviation: " + str(stats[1]))
    print("Kurtosis: " + str(stats[2]))
    print("Sample (N): " + str(stats[3]))
    print("Actual Kurtosis: " + str(stats[4]))
    print("Skewness: " + str(stats[5]))
    print("Jarque-Bera Test: " + str(stats[6]))
    print("Jarque-Bera p-value: " + str(stats[7]))
