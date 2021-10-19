import collections
from stockStatistics import stockReturnsList
from tb3ms import TB3MS_RiskFree

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def portfolioActualReturns(tickerList):
    returns = stockReturnsList(tickerList)

    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)

    for i in tickers:
        returns[i] = returns[i] - riskFree

    return returns


if __name__=="__main__":
    print(portfolioActualReturns(tickers))