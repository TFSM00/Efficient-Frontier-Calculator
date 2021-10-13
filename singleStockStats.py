import pandas as pd
import yahoo_fin.stock_info as yf
import numpy as np
import statistics as stats
import scipy.stats as st

ticker = "GOOG"
data = yf.get_data(ticker, start_date="08/01/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY

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

if "__name__" == "__main__": #if script runs as program not import
    print("Average Returns: "+str(average))
    print("Standard Deviation: "+str(standardDeviation))
    print("Kurtosis: "+str(kurtosis))
    print("Sample (N): "+str(sample))
    print("Actual Kurtosis: "+str(actualKurtosis))
    print("Skewness: "+str(skewness))
    print("Jarque-Bera Test: "+str(jarque_bera_test))
    print("Jarque-Bera p-value: "+str(jarque_bera_p_value))
