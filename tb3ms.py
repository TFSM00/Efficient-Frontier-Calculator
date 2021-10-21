import requests
import pandas as pd 
import numpy as np
import statistics as stats


def TB3MS_Data():
    startdate="2004-08-31"
    enddate="2019-10-31"

    excel_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=748&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=TB3MS&scale=left&cosd={startdate}&coed={enddate}&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2021-10-10&revision_date=2021-10-10&nd=1934-01-01"
    r = requests.get(excel_url)

    with open("treasurybill.csv",'wb') as f:
        f.write(r.content)
        f.close()

    tb3ms_data = pd.read_csv("treasurybill.csv")
    tb3ms_perc = tb3ms_data["TB3MS"].tolist()

    tb3ms = []
    effective_monthly_rate = []

    for i in range(0, len(tb3ms_perc)):
        tb3ms.append(round(float(tb3ms_perc[i]/100),ndigits=4))

    for i in range(0, len(tb3ms)):
        effective_monthly_rate.append(float((((1+(tb3ms[i]/4)))**(1/3))-1))

    average = np.mean(effective_monthly_rate)
    stdev = stats.stdev(effective_monthly_rate)

    return {"Average":average,"Standard Deviation": stdev, "Last Effective Monthly Rate": tb3ms[-1]}
    

def TB3MS_RiskFree():
    startdate="2004-08-31"
    enddate="2019-10-31"

    excel_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=748&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=TB3MS&scale=left&cosd={startdate}&coed={enddate}&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2021-10-10&revision_date=2021-10-10&nd=1934-01-01"
    r = requests.get(excel_url)

    with open("treasurybill.csv",'wb') as f:
        f.write(r.content)
        f.close()

    tb3ms_data = pd.read_csv("treasurybill.csv")
    tb3ms_perc = tb3ms_data["TB3MS"].tolist()

    tb3ms = []
    for i in range(0, len(tb3ms_perc)):
        tb3ms.append(round(float(tb3ms_perc[i]/100),ndigits=4))

    return tb3ms


if __name__=="__main__":
    print(TB3MS_Data())





