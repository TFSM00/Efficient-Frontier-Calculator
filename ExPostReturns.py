from stockStatistics import stockReturnsList

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]



a = stockReturnsList(tickers)
halfLen = int(len(a[tickers[0]])/2)
b = a.drop(a.index[:halfLen])

equalws = [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]



for i in range(0,len(tickers)):
    for j in range(0,len(tickers)):
        item = b.iloc[i,j]
        new_item = float(item * equalws[i])
        b.iloc[i,j]=new_item




print(b)