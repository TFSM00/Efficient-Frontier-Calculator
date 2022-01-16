import numpy as np
import pandas_datareader as pdr
import datetime as dt 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dateutil import relativedelta as rd
from scipy import stats
import statsmodels.api as sm
from statsmodels import regression
from sklearn.linear_model import LinearRegression
from statsmodels.regression.rolling import RollingOLS
import getFamaFrenchFactors as gff

ff3 = gff.famaFrench3Factor(frequency="m")
ff3 = ff3.set_index("date_ff_factors")
riskfree = ff3["RF"][-1]
print(ff3)
print(riskfree)



