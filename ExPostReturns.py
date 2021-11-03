from stockStatistics import stockReturnsList
from tb3ms import TB3MS_RiskFree
import numpy as np


tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]



a = stockReturnsList(tickers)


equalws = [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]

b = a 

for i in range(0,len(tickers)):
    for j in range(0,len(tickers)):
        item = b.iloc[i,j]
        new_item = float(item * equalws[i])
        b.iloc[i,j]=new_item


dflenght = len(b)
exPost = []

for i in range(0, dflenght):
    sumProd = 0
    row = list(a.iloc[i])
    for j in row:
        ind = row.index(j) 
        sumProd += float(j * equalws[ind])
    
    exPost.append(sumProd)

a["Ex-Post Returns"] = exPost

riskFree = TB3MS_RiskFree()
riskFree.pop(0)

exPostActualReturns = np.subtract(exPost - riskFree)

a["Ex-Post Actual Returns"] = exPostActualReturns



print(a)

