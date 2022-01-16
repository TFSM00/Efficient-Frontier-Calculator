import yahoo_fin.stock_info as yf

tick = "^GSPC"

def getStockPrice(ticker):
    """
    Gets the raw prices data as a dataframe
    """
    data = yf.get_data(ticker, start_date="08/31/2004",end_date="10/31/2019", interval="1mo") # DATE IS MM/DD/YYYY

    return data

if __name__=="__main__":
    print(getStockPrice(tick))