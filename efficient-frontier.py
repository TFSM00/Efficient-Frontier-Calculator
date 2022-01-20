import numpy as np
import pandas_datareader as pdr
import datetime as dt 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dateutil import relativedelta as rd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.regression.rolling import RollingOLS
import getFamaFrenchFactors as gff

def ticker_check(tickers, start_date, end_date):
    res = []
    for i in tickers:
        try:
            data = pdr.get_data_yahoo(i, start_date, end_date)
            res.append(True)
        except:
            res.append(False)
    
    if all(res):
        return True
    return False
        


def riskFreeRate():
    ff3 = gff.famaFrench3Factor(frequency="m")
    ff3 = ff3.set_index("date_ff_factors")
    riskfree = ff3["RF"][-1]
    return riskfree

def beta_rolling(tickers, start_date, end_date, window):
    new_tickers = tickers
    new_tickers.append("^GSPC")

    data = pdr.get_data_yahoo(new_tickers, start_date, end_date, interval="m")
    data = data["Adj Close"]

    log_returns = np.log(data/data.shift())

    betas = pd.DataFrame()

    for i in tickers:
        Y = log_returns[i]
        X = log_returns["^GSPC"]
                
        x = sm.add_constant(X)
        rols = RollingOLS(Y,x, window)
        rres = rols.fit()
        params = rres.params.copy()
        betas[i] = params["^GSPC"]

    del betas["^GSPC"]


    fig, ax = plt.subplots()
    if len(list(betas.columns))==1:
        plt.plot(betas, label= list(betas.columns)[0])
    else:
        plt.plot(betas, label=list(betas.columns))
    ax.legend()
    ax.set(title="Rolling Beta", xlabel="Date", ylabel="Beta")
    
    return fig


def correlation(tickers, start_date, end_date):
    new_tickers = tickers
    new_tickers.append("^GSPC")
    

    data = pdr.get_data_yahoo(tickers, start_date, end_date, interval="m")

    data = data["Adj Close"]
    
    log_returns = np.log(data/data.shift())
    correlation = log_returns.corr()
    corr = pd.DataFrame(correlation)

    corr = corr.rename(columns = {"^GSPC":"S&P500"})
    corr = corr.T
    corr = corr.rename(columns = {"^GSPC":"S&P500"})

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True)
    ax.set(title="Correlation Matrix", xlabel="Tickers", ylabel="Tickers")
    

    return fig

def log_rets(tickers, start_date, end_date):
    data = pdr.get_data_yahoo(tickers, start_date, end_date, interval="m")
    data = data["Adj Close"]
    
   

    log_returns = np.log(data/data.shift())
    return log_returns

def efficientFrontier(tickers, start_date, end_date):

    data = pdr.get_data_yahoo(tickers, start_date, end_date, interval="m")
    data = data["Adj Close"]

    log_returns = np.log(data/data.shift())

    weight = np.random.random(len(tickers))
    weight /= weight.sum()

    #exp_return = np.sum(log_returns.mean()* weight)*252

    #exp_vol = np.sqrt(np.dot(weight, np.dot(log_returns.cov()*252,weight)))

    rf = riskFreeRate()

    n = 5000 #runs
    weights = np.zeros((n,len(tickers)))
    exp_rtns = np.zeros(n)
    exp_vols = np.zeros(n)
    sharpe_ratios = np.zeros(n)

    for i in range(n):
        weight = np.random.random(len(tickers))
        weight /= weight.sum()
        weights[i] = weight

        exp_rtns[i] = np.sum(log_returns.mean()*weight)*252
        exp_vols[i] = np.sqrt(np.dot(weight, np.dot(log_returns.cov()*252,weight)))
        sharpe_ratios[i] = (exp_rtns[i]-rf)/exp_vols[i]

    fig,ax = plt.subplots()

    main = ax.scatter(exp_vols,exp_rtns, c=sharpe_ratios)
    ax.scatter(exp_vols[sharpe_ratios.argmax()], exp_rtns[sharpe_ratios.argmax()], c="red", label="Highest Sharpe Ratio Portfolio")
    ax.scatter(exp_vols[exp_vols.argmin()], exp_rtns[exp_vols.argmin()], c="orange", label="Minimum Risk Portfolio")
    ax.legend()
    #ax.set_ylim(0)
    ax.set(title="Efficient Frontier")
    ax.set_xlabel("Expected Volatility")
    ax.set_ylabel("Expected Return")
    fig.colorbar(main, label="Sharpe Ratio")

    return fig


def beta(tickers, start_date, end_date):
    new_tickers = tickers
    new_tickers.append("^GSPC")

    data = pdr.get_data_yahoo(new_tickers, start_date, end_date, interval="m")
    data = data["Adj Close"]

    log_returns = np.log(data/data.shift())
    # log_returns2 = pd.DataFrame([log_returns["AAPL"], log_returns["^GSPC"]])
    # log_returns2 = log_returns2.T
    # log_returns2 = log_returns2[1:]    
    data = pd.DataFrame(index=["Alpha","Beta"])
    
    for i in tickers:
        log_returns2 = pd.DataFrame([log_returns[i],log_returns["^GSPC"]])
        log_returns2 = log_returns2.T
        log_returns2 = log_returns2.dropna()

        X = log_returns2[i].iloc[1:].to_numpy().reshape(-1,1) #remove first item (NaN), model doesnt like Nan nor DataFrames
        Y = log_returns2["^GSPC"].iloc[1:].to_numpy().reshape(-1,1)

        lin_regr = LinearRegression()
        lin_regr.fit(Y,X)
        Y_pred = lin_regr.predict(X)
        alpha = lin_regr.intercept_[0]
        beta = lin_regr.coef_[0,0]

        data[i] = [alpha, beta]
        
    del data["^GSPC"]
    return data
    

def securityMarketLine(tickers, start_date, end_date):
    beta_table = beta(tickers, start_date, end_date).T
    risk_free = riskFreeRate()

    ff3 = gff.famaFrench3Factor(frequency="m")
    ff3 = ff3.set_index("date_ff_factors")
    mkt_premium = ff3["Mkt-RF"][-1]

    beta_table["Expected Return"] = beta_table["Alpha"] + (mkt_premium * beta_table["Beta"])
    del beta_table["Alpha"]
    c = [i for i in range(0, len(beta_table["Beta"]))]
    fig, ax = plt.subplots()
    ticks = list(beta_table.index)
    for i in ticks:
        ax.scatter(beta_table["Beta"][i], beta_table["Expected Return"][i], label=i)
    
    sml = ax.plot([0, beta_table["Beta"].max()+0.2],[risk_free, 2 * mkt_premium + risk_free ], label="SML", c="red")
    ax.axhline(0,color='black') # x = 0
    ax.axvline(0,color='black') # y = 0
    ax.legend(loc="upper left")
    ax.set(title="Security Market Line")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Expected Return")


    return fig

#st.set_page_config(layout="wide")

sidebar_title = st.sidebar.header("Portfolio Efficient Frontier")
author = st.sidebar.write("Made by Tiago Moreira")
space = st.sidebar.header("")
ticker_input = st.sidebar.text_input("Enter the tickers space-separated:")
tickers = ticker_input.strip()
tickers = tickers.split(" ")

while("" in tickers):
    tickers.remove("")

    
start_date = st.sidebar.date_input('Select a starting date:', min_value=dt.datetime(1950,1,1))
end_date = st.sidebar.date_input('Select an ending date:', max_value=dt.datetime.today())
month_delta = rd.relativedelta(end_date,start_date).years * 12
beta_window = st.sidebar.slider("Beta rolling window (in months)", 1, int(month_delta/4))
run_button = st.sidebar.button("Run calculations")


with st.spinner(text='In progress - wait for calculations to complete in order to scroll down'):
    if run_button:
        if not ticker_check(tickers, start_date, end_date):
            st.sidebar.error("Error: one of the tickers used does not exist or was misspelled")
        else:
            if start_date < end_date:
                #st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
                if month_delta<12:
                    st.sidebar.warning("Warning: at least a year of data is necessary for more accurate calculations")
                st.subheader("Correlation Matrix")
                st.pyplot(correlation(tickers, start_date, end_date))
                st.subheader("Efficient Frontier")
                st.pyplot(efficientFrontier(tickers, start_date, end_date))  #add capital market line radio button
                st.subheader("Alpha and Beta Statistics")
                st.table(beta(tickers, start_date, end_date))
                st.subheader("Security Market Line")
                st.pyplot(securityMarketLine(tickers, start_date, end_date))
                st.subheader("Rolling Beta")
                st.pyplot(beta_rolling(tickers,start_date, end_date, beta_window))
            else:
                st.sidebar.error('Error: End date must fall after start date')