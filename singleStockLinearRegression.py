import pandas as pd
from portfolioActualReturns import portfolioActualReturns
from marketRiskPremium import marketActualReturns
import statsmodels.api as sm

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def singleStockLinearRegression_Summary(tickerList, ind):
    
    actual_returns = portfolioActualReturns(tickerList) #the stock has to be in the list
    market = marketActualReturns()

    ticker = tickerList[ind]
    stockReturns = actual_returns[ticker]

    data={ticker: stockReturns, "S&P500":market}
    df = pd.DataFrame.from_dict(data)

    Y = df[ticker]
    X=df["S&P500"]

    X = sm.add_constant(X)

    model = sm.OLS(Y,X).fit()

    summary = model.summary()

    return summary



def singleStockLinearRegression_Stats(tickerList, ind):
    actual_returns = portfolioActualReturns(tickerList)
    market = marketActualReturns()
    
    ticker = tickerList[ind]
    
    stockReturns = actual_returns[ticker]

    data={ticker: stockReturns, "S&P500":market}
    df = pd.DataFrame.from_dict(data)

    Y = df[ticker]
    X=df["S&P500"]

    X = sm.add_constant(X)

    model = sm.OLS(Y,X).fit()

    alpha_beta = dict(model.params)

    alpha_beta["Alpha"] = alpha_beta.pop("const")
    alpha_beta["Beta"] = alpha_beta.pop("S&P500")

    p_values = dict(model.pvalues)
    p_values["Alpha p-value"] = p_values.pop("const")
    p_values["Beta p-value"] = p_values.pop("S&P500")

    r_squared = model.rsquared

    return [alpha_beta,p_values,r_squared]

def singleStockLinearRegression_StatsOutput(tickerList, ind): #wont accept just the stock
    actual_returns = portfolioActualReturns(tickerList)
    market = marketActualReturns()

    ticker = tickerList[ind]

    stockReturns = actual_returns[f"{ticker}"]

    data={f"{ticker}": stockReturns, "S&P500":market}
    df = pd.DataFrame.from_dict(data)

    Y = df[f"{ticker}"]
    X=df["S&P500"]

    X = sm.add_constant(X)

    model = sm.OLS(Y,X).fit()

    alpha_beta = dict(model.params)

    alpha_beta["Alpha"] = alpha_beta.pop("const")
    alpha_beta["Beta"] = alpha_beta.pop("S&P500")

    p_values = dict(model.pvalues)
    p_values["Alpha p-value"] = p_values.pop("const")
    p_values["Beta p-value"] = p_values.pop("S&P500")

    r_squared = model.rsquared

    print(f"Linear Regression Stats for {ticker}:")
    print("============================"+"="*len(ticker)+"=")
    print(f"Alpha = {alpha_beta.get('Alpha')}")
    print(f"Alpha p-value = {p_values.get('Alpha p-value')}")
    print(f"Beta = {alpha_beta.get('Beta')}")
    print(f"Beta p-value = {p_values.get('Beta p-value')}")
    print(f"R-Squared = {r_squared}")
    print("============================"+"="*len(ticker)+"=")

    

if __name__=="__main__":
    singleStockLinearRegression_StatsOutput(tickers,0)