import pandas as pd
from portfolioActualReturns import portfolioActualReturns
from marketRiskPremium import marketActualReturns
import statsmodels.api as sm
from tb3ms import TB3MS_Data
from marketRiskPremium import marketAverageRiskPremium
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.colors import ListedColormap

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def portfolioLinearRegression(tickerList):
    actual_returns = portfolioActualReturns(tickerList)
    market_actual_returns = marketActualReturns()
    LinearRegression = pd.DataFrame(columns=["Alpha","Alpha p-value","Beta","Beta p-value","R-Squared"])
    
    for i in tickerList:
        act_ret = actual_returns[i]
        market = market_actual_returns["^GSPC"]

        data={i: act_ret, "S&P500":market}
        df = pd.DataFrame.from_dict(data)

        Y = df[i]
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

        statsList = [alpha_beta.get("Alpha"),p_values.get("Alpha p-value"), alpha_beta.get("Beta"),p_values.get("Beta p-value"), r_squared]
        LinearRegression.loc[i] = statsList
    
    return LinearRegression

def securityBetaTable(tickerList):
    linearRegressionData = portfolioLinearRegression(tickerList)
    betas = linearRegressionData["Beta"].to_list()
    alphas = linearRegressionData["Alpha"].to_list()
    tbData = TB3MS_Data()
    averageRf = tbData.get("Average")
    averageRiskPremium = marketAverageRiskPremium()
    expectedReturn = []
    
    for i in range(0,len(tickerList)):
        Eri = alphas[i]*(betas[i]*averageRiskPremium)
        expectedReturn.append(Eri)


    betaTable = pd.DataFrame(columns=tickerList)
    betaTable.loc["Beta"] = betas
    betaTable.loc["Expected Return"] = expectedReturn
    betaTable["Risk Free"] = [0,averageRf]
    betaTable["Market"] = [2,averageRiskPremium]

    return betaTable


def SecurityMarketLine(tickerList):
    linearRegressionData = portfolioLinearRegression(tickerList)
    betas = linearRegressionData["Beta"].to_list()
    alphas = linearRegressionData["Alpha"].to_list()
    tbData = TB3MS_Data()
    averageRf = tbData.get("Average")
    averageRiskPremium = marketAverageRiskPremium()
    expectedReturn = []
    
    for i in range(0,len(tickerList)):
        Eri = alphas[i]*(betas[i]*averageRiskPremium)
        expectedReturn.append(Eri)
    
    selectList = []
  
    assets = [tick for tick in tickerList]
    
    for i in range(0, len(tickerList)):
        selectList.append(i)
        
    scatter = plt.scatter(betas, expectedReturn, c=selectList)

    dots = zip(betas,expectedReturn)
    iter = 0
    for beta, eri in dots:
        label = tickerList[iter]
        plt.annotate(label, (beta,eri), textcoords = "offset points",xytext=(0,10),ha="center")
        iter+=1
    
    plt.plot([2,averageRiskPremium],[0,averageRf],color="blue")
    plt.legend(handles=scatter.legend_elements()[0], labels = assets)
    plt.show()
    


print(portfolioLinearRegression(tickers))

#print(securityBetaTable(tickers))

#SecurityMarketLine(tickers)

