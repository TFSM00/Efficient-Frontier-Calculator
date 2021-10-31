from stockStatistics import stockReturnsforSingle
from tb3ms import TB3MS_RiskFree
import numpy as np

def marketActualReturns():
    """
    Returns a list with the actual market returns
    """
    
    returns_raw = stockReturnsforSingle("^GSPC")
    returns = returns_raw["^GSPC"].to_list()


    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)


    riskPremium = []
    for item1,item2 in zip(returns,riskFree):
        riskPremium.append(float(item1) - float(item2))

    return riskPremium

def marketAverageRiskPremium():
    """
    Returns the average value for risk premium (market actual returns)
    """
    
    returns_raw = stockReturnsforSingle("^GSPC")
    returns = returns_raw["^GSPC"].to_list()

    riskFree = TB3MS_RiskFree()
    riskFree.pop(0)


    riskPremium = []
    for item1,item2 in zip(returns,riskFree):
        riskPremium.append(float(item1) - float(item2))
    

    return np.mean(riskPremium)


if __name__=="__main__":
    #print(marketActualReturns())
    print(marketAverageRiskPremium())