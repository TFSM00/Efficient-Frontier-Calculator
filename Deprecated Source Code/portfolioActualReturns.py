from stockStatistics import stockReturnsforSingle, stockReturnsList
from tb3ms import TB3MS_RiskFree

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def portfolioActualReturns(tickerList):
    """
    Returns a dataframe with the actual returns for each stock
    Actual Returns = Stock Returns - Risk Free Rate
    """

    returns = stockReturnsList(tickerList)

    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)

    for i in tickers:
        returns[i] = returns[i]- riskFree

    return returns
    

def singleStockActualReturns(ticker):
    """
    Returns a list with the actual returns for a single stock
    """
    
    returns_raw = stockReturnsforSingle(ticker)
    returns = returns_raw[ticker].to_list()

    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)

    for i in range(0,len(returns)):
        returns[i] = returns[i] - riskFree[i]

    return returns


if __name__=="__main__":
    print(portfolioActualReturns(tickers))