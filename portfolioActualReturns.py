from stockStatistics import stockReturnsforSingle, stockReturnsList

from tb3ms import TB3MS_RiskFree


tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def portfolioActualReturns(tickerList):
    returns = stockReturnsList(tickerList)

    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)

    for i in tickers:
        returns[i] = returns[i]- riskFree

    return returns
    

def singleStockActualReturns(ticker):
    returns_raw = stockReturnsforSingle(ticker)
    returns = returns_raw[ticker].to_list()

    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)

    for i in range(0,len(returns)):
        returns[i] = returns[i] - riskFree[i]

    return returns


if __name__=="__main__":
    print(portfolioActualReturns(tickers))