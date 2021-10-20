from stockStatistics import stockReturnsList
from tb3ms import TB3MS_RiskFree
import numpy as np

def marketActualReturns():
    returns = stockReturnsList(["^GSPC"])

    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)


    returns["^GSPC"] = returns["^GSPC"] - riskFree

    return returns

def marketAverageRiskPremium():
    returns = stockReturnsList(["^GSPC"])

    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)


    riskPremium = returns["^GSPC"] - riskFree

    return np.mean(riskPremium)


if __name__=="__main__":
    print(marketActualReturns())