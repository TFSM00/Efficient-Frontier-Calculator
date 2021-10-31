import pandas as pd
from portfolioActualReturns import singleStockActualReturns
from marketRiskPremium import marketActualReturns
import statsmodels.api as sm


tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def singleStockLinearRegression_Summary(ticker):
    """
    Returns a summary for the Linear Regression of a single stock, just like in excel
    """
    
    stockReturns = singleStockActualReturns(ticker) 
    market = marketActualReturns()

    data={ticker: stockReturns, "S&P500":market}
    df = pd.DataFrame.from_dict(data)

    Y = df[ticker]
    X= df["S&P500"]

    X = sm.add_constant(X)

    model = sm.OLS(Y,X).fit() # Regression Model

    summary = model.summary()

    return summary


def singleStockLinearRegression_Stats(ticker):
    """
    Returns the regression stats as a list for a single ticker. Stats are Alpha, Beta, p-values for each and R^2
    """
    stockReturns = singleStockActualReturns(ticker)
    market = marketActualReturns()
    
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

def singleStockLinearRegression_StatsOutput(ticker): 
    """
    Displays the statistical results explained above
    """

    stockReturns = singleStockActualReturns(ticker)
    market = marketActualReturns()


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
    singleStockLinearRegression_StatsOutput("INTC")
    #print(singleStockLinearRegression_Summary("INTC"))
    #print(singleStockLinearRegression_Stats("INTC"))