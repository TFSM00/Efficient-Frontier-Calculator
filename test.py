


        
tickers = ["AAPL","MSFT","IBM"]
tickers.append("^GSPC")

print(tickers)
import pandas_datareader as pdr
import datetime as dt 

start_date=dt.datetime(2007,12,12)
end_date = dt.datetime(2022,1,22)

data = pdr.get_data_yahoo(tickers, start_date, end_date, interval="m")

    
data = data["Adj Close"]

print(data)
