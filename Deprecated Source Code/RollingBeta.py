from stockStatistics import stockReturnsforSingle
from marketData import marketReturnswithDates
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

tickers = ["AAPL","GOOG","AMZN","MSFT","INTC","IBM","ORCL","CSCO","NVDA"]

def RollingBetaGraphs(tickerList):
    """
    Opens a pdf and displays the rolling beta for every ticker in the list
    """
    
    with PdfPages("RollingBeta.pdf") as pdf:
        for tick in tickerList:  
            stockRet = stockReturnsforSingle(tick)
            mktRet = marketReturnswithDates()

            stockRet["S&P500"] = mktRet["S&P500"]

            betas = [None]*36
            stddevs = [None]*36
            UpperBound95 = [None]*36
            LowerBound95 = [None]*36
            const = 2.03224450931772


            for i in range(0, len(stockRet["S&P500"])-36):
                tickRet = stockRet[tick][i:i+36]
                marketRet = mktRet["S&P500"][i:i+36]
                beta, stddev = np.polyfit(marketRet,tickRet,1)
                betas.append(beta)
                stddevs.append(stddev)
                UpperBound95.append(float(beta + (const*stddev)))
                LowerBound95.append(float(beta - (const*stddev)))


            stockRet[f"{tick} Beta"]=betas
            stockRet["Standard Deviation"]=stddevs
            stockRet[r"95% Lower Bound"]=LowerBound95
            stockRet[r"95% Upper Bound"]=UpperBound95

            newTable = stockRet

            del newTable[tick]
            del newTable["S&P500"]
            del newTable["Standard Deviation"]
            newTable = newTable[36:]

            plt.figure()
            plt.plot(newTable)
            plt.title(f"Rolling Beta for {tick}")
            plt.xlabel("Date")
            plt.ylabel("Rolling Beta")
            plt.grid()
            plt.legend([f"{tick} Beta",r"95% Lower Bound", r"95% Upper Bound"])
            pdf.savefig() # Adds plot to the pdf
            
if __name__ == "__main__":       
    RollingBetaGraphs(tickers)



