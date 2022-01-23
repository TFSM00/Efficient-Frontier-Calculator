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
#import time
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
        
def riskFreeRate():
    ff3 = gff.famaFrench3Factor(frequency="m")
    ff3 = ff3.set_index("date_ff_factors")
    riskfree = ff3["RF"][-1]
    mkt_premium = ff3["Mkt-RF"][-1]

    return riskfree, mkt_premium

def get_returns(tickers, start_date, end_date):
    temp_ticks = tickers.copy()
    temp_ticks.append("^GSPC")
    try:
        data = pdr.get_data_yahoo(temp_ticks, start_date, end_date, interval="m")
    except:
        err = pd.DataFrame()
        return err
    
    data = data["Adj Close"]

    log_returns = np.log(data/data.shift())
    return log_returns

def correlation(log_rets):
    log_returns_corr = log_rets.copy()
    correlation = log_returns_corr.corr()
    corr = pd.DataFrame(correlation)

    corr = corr.rename(columns = {"^GSPC":"S&P500"})
    corr = corr.T
    corr = corr.rename(columns = {"^GSPC":"S&P500"})

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True)
    ax.set(title="Correlation Matrix", xlabel="Tickers", ylabel="Tickers")
    

    return fig

def efficientFrontier(log_rets, rf):
    log_returns_ef = log_rets.copy()
    del log_returns_ef["^GSPC"]
    n = 5000 #runs
    weights = np.zeros((n,len(input_tickers)))
    exp_rtns = np.zeros(n)
    exp_vols = np.zeros(n)
    sharpe_ratios = np.zeros(n)

    for i in range(n):
        weight = np.random.random(len(input_tickers))
        weight /= weight.sum()
        weights[i] = weight

        exp_rtns[i] = np.sum(log_returns_ef.mean()*weight)*12*(252/365)
        exp_vols[i] = np.sqrt(np.dot(weight.T, np.dot(log_returns_ef.cov()*12*(252/365),weight)))
        sharpe_ratios[i] = (exp_rtns[i]-rf)/exp_vols[i]

    
    ef_fig,ax = plt.subplots()

    main = ax.scatter(exp_vols,exp_rtns, c=sharpe_ratios)
    ax.scatter(exp_vols[sharpe_ratios.argmax()], exp_rtns[sharpe_ratios.argmax()], c="red", label="Highest Sharpe Ratio Portfolio")
    ax.scatter(exp_vols[exp_vols.argmin()], exp_rtns[exp_vols.argmin()], c="orange", label="Minimum Risk Portfolio")
    ax.legend(prop={'size': 8})
    #ax.set_ylim(0)
    ax.set(title="Efficient Frontier")
    ax.set_xlabel("Expected Volatility")
    ax.set_ylabel("Expected Return")
    ef_fig.colorbar(main, label="Sharpe Ratio")
    plt.grid()

    return ef_fig

def beta(log_rets):
    log_returns_beta = log_rets.copy()
    data = pd.DataFrame(index=["Alpha","Beta"])
    
    for i in input_tickers:
        log_returns2 = pd.DataFrame([log_returns_beta[i],log_returns_beta["^GSPC"]])
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
        
    return data

def securityMarketLine(log_rets, rf, mkt_premium):
    beta_table = beta(log_rets.copy()).T

    beta_table["Expected Return"] = beta_table["Alpha"] + (mkt_premium * beta_table["Beta"])
    del beta_table["Alpha"]
    c = [i for i in range(0, len(beta_table["Beta"]))]
    fig, ax = plt.subplots()
    ticks = list(beta_table.index)
    for i in ticks:
        ax.scatter(beta_table["Beta"][i], beta_table["Expected Return"][i], label=i)
    
    sml = ax.plot([0, beta_table["Beta"].max()+0.2],[rf, 2 * mkt_premium + rf ], label="SML", c="red")
    ax.axhline(0,color='black') # x = 0
    ax.axvline(0,color='black') # y = 0
    ax.legend(loc="upper left", prop={'size': 8})
    ax.set(title="Security Market Line")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Expected Return")
    plt.grid()

    return fig

def beta_rolling(log_rets, window):
    log_returns_br = log_rets.copy()

    betas = pd.DataFrame()

    for i in input_tickers:
        Y = log_returns_br[i]
        X = log_returns_br["^GSPC"]
                
        x = sm.add_constant(X)
        rols = RollingOLS(Y,x, window)
        rres = rols.fit()
        params = rres.params.copy()
        betas[i] = params["^GSPC"]


    fig, ax = plt.subplots()
    if len(list(betas.columns))==1:
        plt.plot(betas, label= list(betas.columns)[0])
    else:
        plt.plot(betas, label=list(betas.columns))
    years = YearLocator()   # every year
    months = MonthLocator()  # every month
    yearsFmt = DateFormatter('%Y')
    ax.legend(prop={'size': 8})
    ax.set(title="Rolling Beta", xlabel="Date", ylabel="Beta")
    ax = plt.gca()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.autoscale_view()

    plt.grid()
    return fig



sidebar_title = st.sidebar.header("Portfolio Efficient Frontier")
author = st.sidebar.write("Made by Tiago Moreira")
space = st.sidebar.header("")
ticker_input = st.sidebar.text_input("Enter the tickers space-separated:")
input_tickers = ticker_input.strip()
input_tickers = input_tickers.split(" ")

while("" in input_tickers):
    input_tickers.remove("")

    
start_date = st.sidebar.date_input('Select a starting date:', min_value=dt.datetime(1950,1,1))
end_date = st.sidebar.date_input('Select an ending date:', max_value=dt.datetime.today())
month_delta = rd.relativedelta(end_date,start_date).years * 12
beta_window = st.sidebar.slider("Beta rolling window (in months)", 2, int(month_delta/2))
run_button = st.sidebar.button("Run calculations")

data = get_returns(input_tickers, start_date, end_date)
rf, mkt_premium = riskFreeRate()


with st.spinner(text='In progress - wait for calculations to complete in order to scroll down'):
    if run_button:
        #timer_s = time.time()
        if data.empty:
            st.sidebar.error("Error: one of the tickers used does not exist or was misspelled")
        else:
            if start_date < end_date:
                #st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
                if month_delta<12:
                    st.sidebar.warning("Warning: at least a year of data is necessary for more accurate calculations")
                # st.subheader("Correlation Matrix")
                # st.pyplot(correlation(data))
                # st.subheader("Efficient Frontier")
                # st.pyplot(efficientFrontier(data, rf))  #add capital market line radio button
                # st.subheader("Alpha and Beta Statistics")
                # st.table(beta(data))
                # st.subheader("Security Market Line")
                # st.pyplot(securityMarketLine(data, rf, mkt_premium))
                st.subheader("Rolling Beta")
                st.pyplot(beta_rolling(data, beta_window))
                #st.sidebar.write(f"{time.time()-timer_s} seconds")
            else:
                st.sidebar.error('Error: End date must fall after start date')
