from stockStatistics import stockReturnsList
from tb3ms import TB3MS_RiskFree
import numpy as np

def marketActualReturns():
    returns = stockReturnsList(["SPY"])

    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)


    returns["SPY"] = returns["SPY"] - riskFree

    return returns

def marketAverageRiskPremium():
    returns = stockReturnsList(["SPY"])

    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)


    riskPremium = returns["SPY"] - riskFree

    return np.mean(riskPremium)


if __name__=="__main__":
    print(marketActualReturns())