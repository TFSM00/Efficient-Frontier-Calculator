from tb3ms import TB3MS_Data
from stockExpectedReturn import Average_StdDev_Data, average
import numpy as np
import pandas as pd
from stockCovariance import stockCovariance
import matplotlib.pyplot as plt

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights
ind_er = average(tickers)
cov_matrix = stockCovariance(tickers)
df = Average_StdDev_Data(tickers)

num_assets = 9
num_portfolios = 10000
for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ind_er)
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    p_vol.append(sd)

data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(df.columns.tolist()):
    data[symbol+' weight'] = [w[counter] for w in p_weights]
portfolios  = pd.DataFrame(data)


plt.figure(figsize=(12.2,12.2))
plt.scatter(portfolios["Volatility"], portfolios["Returns"])
plt.title("Efficient Frontier", fontsize=18)
plt.xlabel("Volatility", fontsize=14)
plt.ylabel("Returns", fontsize=14)
plt.grid()
plt.show()