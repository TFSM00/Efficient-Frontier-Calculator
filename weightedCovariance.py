from stockCovariance import stockCovariance

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def weightedCovarianceEqualWeights(tickerList):
    """
    Returns a table with the covariance between all stocks, assuming the portfolio is weighted equally through all stocks
    """
    
    cov = stockCovariance(tickerList) #returns the covariance table
    weights = []
    
    for i in range(0,len(tickerList)): #gets the weight of every single stock
        weights.append(float(1/len(tickerList)))

    for i in range(0,len(tickerList)): #gets every cell in the table and multiplies it by the weights of each pair of stocks
        for j in range(0,len(tickerList)):
            item = cov.iloc[i,j]
            new_item = item * weights[i] * weights[j]
            cov.iloc[i,j]=new_item

    return cov


def weightedCovariance(tickerList,weightsList):
    """
    Returns a table with the covariance but the weights are picked by the user
    """

    if len(tickerList) != len(weightsList): #ckecks for same lenght
        return print("The number of tickers and number of weights are different")
    elif round(sum(weightsList),8) != round(1,8): #ckecks if the weights equal 100%
        return print("Weights do not equal 100%")
    else:

        cov = stockCovariance(tickerList)
    

        for i in range(0,len(tickerList)):
            for j in range(0,len(tickerList)):
                item = cov.iloc[i,j]
                new_item = item * weightsList[i] * weightsList[j]
                cov.iloc[i,j]=new_item

        return cov


if __name__ == "__main__":
    weightedCovarianceEqualWeights(tickers)