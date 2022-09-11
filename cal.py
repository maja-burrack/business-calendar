
import pandas as pd
import numpy as np
import holidays
from pandas.tseries.offsets import CustomBusinessMonthEnd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta, TH, MO

start = '2022-03-10'
end = '2022-12-01'

def create_dates_df(startdate, enddate):
    dates = pd.date_range(start=startdate, end=enddate)
    df = pd.DataFrame(dates)
    df.rename({0: 'date'}, axis=1, inplace=True)
    return df

def add_columns(df):
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['weeknum'] = df['date'].dt.isocalendar().week
    df['day'] = df['date'].dt.day
    return df

def get_holiday_list(df):
    """gets Danish holidays"""
    startdate = np.min(df['date']).year
    enddate = np.max(df['date']).year
    years = [*range(startdate, enddate +1, 1)]
    holidaylist = list()
    for holiday in holidays.Denmark(years=years).items():
        holidaylist.append(holiday)
        
    holidays_df = pd.DataFrame(holidaylist, columns=['date', 'holiday'])
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    
    df = pd.merge(df, holidays_df, on=['date'], how='left')
    return holidaylist

def add_holiday_cols(df):
    holidays = get_holiday_list(df)
    holidays_df = pd.DataFrame(holidays, columns=['date', 'holiday'])
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    df = pd.merge(df, holidays_df, on=['date'], how='left')
    return df

def add_payday(df):
    holidays = list(zip(*get_holiday_list(df)))[0]
    my_freq = CustomBusinessMonthEnd(holidays=holidays)
    paydays = pd.date_range(start, end, freq=my_freq)
    df['is-payday'] = np.where(df['date'].isin(paydays), 1, 0)
    return df

def add_paymonth(df):
    df['paymonth'] = np.where(df['is-payday']==1, df['month']+1, np.nan)
    df['paymonth'] = df['paymonth'].ffill(axis=0)
    df['paymonth'] = np.where(np.isnan(df['paymonth']), df['month'], df['paymonth'])
    df['paymonth'] = np.where(df['paymonth']==13, 1, df['paymonth'])   
    df['paymonth'] = df['paymonth'].astype(int)                     
    return df

def get_blackfridays(df):
    startdate = np.min(df['date']).year
    enddate = np.max(df['date']).year
    years = [*range(startdate, enddate +1, 1)]
    datelist = [date(year,11,1) for year in years]
    blackfridays = [x + relativedelta(weekday=TH(+4)) + timedelta(days=1) for x in datelist]
    return blackfridays

def add_specialdays(df):
    blackfridays = get_blackfridays(df)
    cybermondays = [bf + relativedelta(weekday=MO) for bf in blackfridays]
    df['specialday'] = np.nan
    df['specialday'] = np.where(df['date'].isin(blackfridays), 'Black Friday', df['specialday'])
    df['specialday'] = np.where(df['date'].isin(cybermondays), 'Cyber Monday', df['specialday'])
    return df

def add_financial_year(df, fy_start_month):
    df['fy'] = np.where(df['month']<fy_start_month, df['year'] % 1000 - 1, df['year'] % 1000)
    df['fy'] = df['fy'].astype(str) + (df['fy']+1).astype(str)
    df['fy'] = df['fy'].astype(int)
    return df
    

def create_calendar(startdate, enddate, fy_start_month=4):
    df = create_dates_df(startdate, enddate)
    df = add_columns(df)
    df = add_financial_year(df, fy_start_month)
    df = add_holiday_cols(df)
    df = add_payday(df)
    df = add_paymonth(df)
    df = add_specialdays(df)
    return df

df = create_calendar(start, end)
