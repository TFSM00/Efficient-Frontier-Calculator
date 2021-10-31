import pandas as pd
from portfolioActualReturns import portfolioActualReturns
from marketRiskPremium import marketActualReturns
import statsmodels.api as sm
from tb3ms import TB3MS_Data
from marketRiskPremium import marketAverageRiskPremium
import matplotlib.pyplot as plt


tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def portfolioLinearRegression(tickerList):
    """
    Returns the Linear Regression Stats as a dataframe for every ticker
    """
    
    actual_returns = portfolioActualReturns(tickerList)
    market_actual_returns = marketActualReturns()
    LinearRegression = pd.DataFrame(columns=["Alpha","Alpha p-value","Beta","Beta p-value","R-Squared"])
    
    for i in tickerList:
        act_ret = actual_returns[i]
        market = market_actual_returns

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
    """
    Returns the betas and expected return for each stock, the t-bill and the market as a dataframe
    """
    
    linearRegressionData = portfolioLinearRegression(tickerList)
    betas = linearRegressionData["Beta"].to_list()
    alphas = linearRegressionData["Alpha"].to_list()
    tbData = TB3MS_Data()
    averageRf = tbData.get("Average")
    averageRiskPremium = marketAverageRiskPremium()
    expectedReturns = []
    
    for i in range(0,len(tickerList)): # Calculation of the expected return
        expRet = alphas[i]*(betas[i]*averageRiskPremium)
        expectedReturns.append(expRet)


    betaTable = pd.DataFrame(columns=tickerList)
    betaTable.loc["Beta"] = betas
    betaTable.loc["Expected Return"] = expectedReturns
    betaTable["Risk Free"] = [0,averageRf]
    betaTable["Market"] = [2,averageRiskPremium]

    return betaTable


def SecurityMarketLine(tickerList):
    """
    Displays the security market line and scatters the betas and expected returns in the plot
    """
    
    linearRegressionData = portfolioLinearRegression(tickerList)
    betas = linearRegressionData["Beta"].to_list()
    alphas = linearRegressionData["Alpha"].to_list()
    tbData = TB3MS_Data()
    averageRf = tbData.get("Average")
    averageRiskPremium = marketAverageRiskPremium()
    expectedReturns = []
    
    for i in range(0,len(tickerList)): # Calculation of the expected return
        expRet = alphas[i]*(betas[i]*averageRiskPremium)
        expectedReturns.append(expRet)
    
    selectList = []
    
    for i in range(0, len(tickerList)):
        selectList.append(i)
        
    scatter = plt.scatter(betas, expectedReturns, c=selectList)

    dots = zip(betas,expectedReturns)
    iter = 0
    for beta, expRet in dots:
        label = tickerList[iter]
        plt.annotate(label, (beta,expRet), textcoords = "offset points",xytext=(0,10),ha="center") # Adds ticker above the dots on the plot
        iter+=1
    
    plt.plot([2,averageRiskPremium],[0,averageRf],color="blue") # Plots the SML
    plt.legend(handles=scatter.legend_elements()[0], labels = tickerList) # Adds a legend to the plot
    plt.axhline(linewidth=1, color="black") # Axis Line
    plt.axvline(linewidth=1, color="black") # Axis Line
    plt.show()
    


#print(portfolioLinearRegression(tickers))

#print(securityBetaTable(tickers))

SecurityMarketLine(tickers)

