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
rf = TB3MS_Data().get("Last Effective Monthly Rate")

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

lenght = len(portfolios["Returns"])
sharpe_ratios = []

for i in range(lenght):
    sharpe_ratios.append((portfolios["Returns"][i]-rf)/portfolios["Volatility"][i])


portfolios["Sharpe Ratio"] = sharpe_ratios
min_vol_port = portfolios.iloc[portfolios["Volatility"].idxmin()]
optimal_risky_port = portfolios.iloc[((portfolios["Returns"]-rf)/portfolios["Volatility"]).idxmax()]

plt.subplots(figsize=[30,20])
main = plt.scatter(portfolios["Volatility"], portfolios["Returns"], c=sharpe_ratios)
plt.scatter(min_vol_port[1],min_vol_port[0], color="r", marker="o", s=100, label="Minimum Volatility Portfolio")
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='o', s=100, label="Optimal Portfolio")
plt.title("Efficient Frontier", fontsize=18)
plt.xlabel("Volatility", fontsize=14)
plt.ylabel("Returns", fontsize=14)

index = sharpe_ratios.index(max(sharpe_ratios))

plt.plot([0, portfolios["Volatility"][index]],[rf, portfolios["Returns"][index]], "-", linewidth=3, color ="green", label = "Capital Market Line")

plt.grid()
plt.colorbar(main)
plt.legend()

plt.show()
