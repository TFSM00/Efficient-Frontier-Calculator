from stockCovariance import stockCovariance

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def weightedCovarianceEqualWeights(tickerList):
    cov = stockCovariance(tickerList)
    weights = []
    
    for i in range(0,len(tickerList)):
        weights.append(float(1/len(tickerList)))

    for i in range(0,len(tickerList)):
        for j in range(0,len(tickerList)):
            item = cov.iloc[i,j]
            new_item = item * weights[i] * weights[j]
            cov.iloc[i,j]=new_item

    return cov


def weightedCovariance(tickerList,weightsList):
    if len(tickerList) != len(weightsList):
        return print("The number of tickers and number of weights are different")
    elif round(sum(weightsList),8) != round(1,8):
        return print("Weights do not equal 100%")
    else:

        cov = stockCovariance(tickerList)
    

        for i in range(0,len(tickerList)):
            for j in range(0,len(tickerList)):
                item = cov.iloc[i,j]
                new_item = item * weightsList[i] * weightsList[j]
                cov.iloc[i,j]=new_item

        return cov



