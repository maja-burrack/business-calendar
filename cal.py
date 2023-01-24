
import pandas as pd
import numpy as np
import holidays
from pandas.tseries.offsets import CustomBusinessMonthEnd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta, TH, MO

start = date(2022,3,10) #'2022-03-10'
end = date(2023,1,3) #'2022-12-01'

def create_dates_df(startdate, enddate):
    dates = pd.date_range(start=startdate, end=enddate)
    df = pd.DataFrame(dates)
    df.rename({0: 'date'}, axis=1, inplace=True)
    return df

def add_standard_date_columns(df):
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['weeknum'] = df['date'].dt.isocalendar().week
    df['weekday'] = df['date'].dt.weekday+1
    df['day'] = df['date'].dt.day
    return df

def add_holiday_cols(df, country='Denmark'):
    """gets holidays"""
    # get list of holidays in country
    startdate = np.min(df['date']).year
    enddate = np.max(df['date']).year
    years = [*range(startdate, enddate +1, 1)]
    holidaylist = list()
    for holiday in getattr(holidays, country)(years=years).items():
        holidaylist.append(holiday)
    
    # create to holiday columns from list of holidays
    holidays_df = pd.DataFrame(holidaylist, columns=['date', 'holiday'])
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    df = pd.merge(df, holidays_df, on=['date'], how='left')
    df['holiday'] = df['holiday'].fillna("")
    return df

def add_payday_columns(df):
    # paydays
    holidays = list(zip(*_get_holiday_list(df)))[0]
    my_freq = CustomBusinessMonthEnd(holidays=holidays)
    paydays = pd.date_range(start, end, freq=my_freq)
    df['is-payday'] = np.where(df['date'].isin(paydays), 1, 0)

    # move months so they start on payday
    df['paymonth'] = np.where(df['is-payday']==1, df['month']+1, np.nan)
    df['paymonth'] = df['paymonth'].ffill(axis=0)
    df['paymonth'] = np.where(np.isnan(df['paymonth']), df['month'], df['paymonth'])
    df['paymonth'] = np.where(df['paymonth']==13, 1, df['paymonth'])   
    df['paymonth'] = df['paymonth'].astype(int)
    # move year accordingly for grouping purposes
    df['payyear'] = np.where(df['paymonth']<df['month'], df['year']+1, df['year'])
    
    # transform days in paymonth to range from 0 to 1
    df['dist_since_payday'] = df.groupby(['payyear', 'paymonth']).cumcount()
    df['dist_max'] = df.groupby(['payyear', 'paymonth'])['dist_since_payday'].transform('max') 
    df['dist_since_payday'] = df['dist_since_payday'] / df['dist_max']
    df.drop('dist_max', axis=1, inplace=True)    
    return df

def add_specialdays(df):
    # BLACKFRIDAYS
    # list of blackfridays
    startdate = np.min(df['date']).year
    enddate = np.max(df['date']).year
    years = [*range(startdate, enddate +1, 1)]
    datelist = [date(year,11,1) for year in years]
    blackfridays = [x + relativedelta(weekday=TH(+4)) + timedelta(days=1) for x in datelist]
    cybermondays = [bf + relativedelta(weekday=MO) for bf in blackfridays]
    blackweekend = [bf + relativedelta(days=1) for bf in blackfridays] \
        + [bf + relativedelta(days=2) for bf in blackfridays]
    
    # create and specialday column with blackfriday-related days    
    df['specialday'] = ""
    df['specialday'] = np.where(df['date'].isin(blackfridays), 'Black Friday', df['specialday'])
    df['specialday'] = np.where(df['date'].isin(cybermondays), 'Cyber Monday', df['specialday'])
    df['specialday'] = np.where(df['date'].isin(blackweekend), 'Black Weekend', df['specialday'])
    
    # TODO: EASTER
    
    return df

def add_financial_year(df, fy_start_month):
    df['fy'] = np.where(df['month']<fy_start_month, df['year'] % 1000 - 1, df['year'] % 1000)
    df['fy'] = df['fy'].astype(str) + (df['fy']+1).astype(str)
    df['fy'] = df['fy'].astype(int)
    return df


def create_calendar(startdate, enddate, fy_start_month=4):
    df = create_dates_df(startdate-relativedelta(days=32), enddate+relativedelta(days=32))
    df = add_standard_date_columns(df)
    df = add_financial_year(df, fy_start_month)
    df = add_holiday_cols(df)
    df = add_payday_columns(df)
    df = add_specialdays(df)
    df = df.loc[(df['date'].dt.date >= startdate) & (df['date'].dt.date <= enddate)]
    return df

def encode_cyclic(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * (df[col]-1)/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * (df[col]-1)/max_val)
    return df

def transform_for_ml(df):
    df['is-holiday'] = np.where(df['holiday']!="", 1, 0)
    df['is-specialday'] = np.where(df['specialday']!="", 1, 0)
    
    # transform cyclical features
    df = encode_cyclic(df, 'month', 12)
    df = encode_cyclic(df, 'paymonth', 12)
    df = encode_cyclic(df, 'weekday', 7)
    
    return df

df = create_calendar(start, end)
df_transformed = transform_for_ml(df)
df_transformed.tail(10)
