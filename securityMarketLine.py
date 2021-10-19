import pandas as pd
from portfolioActualReturns import portfolioActualReturns
from marketRiskPremium import marketActualReturns
import statsmodels.api as sm

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def portfolioLinearRegression(tickerList):
    actual_returns = portfolioActualReturns(tickerList)
    market_actual_returns = marketActualReturns()
    LinearRegression = pd.DataFrame(columns=["Alpha","Alpha p-value","Beta","Beta p-value","R-Squared"])
    
    for i in tickerList:
        act_ret = actual_returns[i]
        market = market_actual_returns["SPY"]

        data={i: act_ret, "SPY":market}
        df = pd.DataFrame.from_dict(data)

        Y = df[i]
        X=df["SPY"]

        X = sm.add_constant(X)

        model = sm.OLS(Y,X).fit()

        alpha_beta = dict(model.params)
        alpha_beta["Alpha"] = alpha_beta.pop("const")
        alpha_beta["Beta"] = alpha_beta.pop("SPY")

        p_values = dict(model.pvalues)
        p_values["Alpha p-value"] = p_values.pop("const")
        p_values["Beta p-value"] = p_values.pop("SPY")

        r_squared = model.rsquared

        statsList = [alpha_beta.get("Alpha"),p_values.get("Alpha p-value"), alpha_beta.get("Beta"),p_values.get("Beta p-value"), r_squared]
        LinearRegression.loc[i] = statsList
    
    return LinearRegression

print(portfolioLinearRegression(tickers))

