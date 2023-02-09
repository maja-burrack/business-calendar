
import pandas as pd
import numpy as np
import holidays
from pandas.tseries.offsets import CustomBusinessMonthEnd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta, TH, MO

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

def get_holiday_list(df, country='Denmark'):
    # get list of holidays in country
    startdate = np.min(df['date']).year
    enddate = np.max(df['date']).year
    years = [*range(startdate, enddate +1, 1)]
    holidaylist = list()
    for holiday in getattr(holidays, country)(years=years).items():
        holidaylist.append(holiday)
    return holidaylist

def add_holiday_cols(df, country='Denmark'):
    """gets holidays"""
    holidaylist = get_holiday_list(df, country)
    
    # create to holiday columns from list of holidays
    holidays_df = pd.DataFrame(holidaylist, columns=['date', 'holiday'])
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    df = pd.merge(df, holidays_df, on=['date'], how='left')
    df['holiday'] = df['holiday'].fillna("")
    df['is-holiday'] = np.where(df['holiday']!="", 1, 0)
    return df
    
def add_payday_columns(df, country='Denmark'):
    # paydays
    holidays = list(zip(*get_holiday_list(df, country)))[0]
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
    df['dist_since_payday'] = df.groupby(['payyear', 'paymonth']).cumcount() # number the days starting on payday
    df['dist_max'] = df.groupby(['payyear', 'paymonth'])['dist_since_payday'].transform('max') # max number of days in paymonth
    df['dist_since_payday'] = df['dist_since_payday'] / df['dist_max'] # make number fall in 0 to 1
    df.drop('dist_max', axis=1, inplace=True)    
    return df

def add_specialdays(df, mindate, maxdate):
    # BLACKFRIDAYS
    # list of blackfridays
    startdate = mindate.year
    enddate = maxdate.year
    years = [*range(startdate, enddate +1, 1)]
    datelist = [date(year,11,1) for year in years]
    blackfridays = [x + relativedelta(weekday=TH(+4)) + timedelta(days=1) for x in datelist]
    cybermondays = [bf + relativedelta(weekday=MO) for bf in blackfridays]
    blackweekend = [bf + relativedelta(days=1) for bf in blackfridays] \
        + [bf + relativedelta(days=2) for bf in blackfridays]
    
    # create a specialday column with blackfriday-related days    
    df['specialday'] = ""
    df['specialday'] = np.where(df['date'].isin(blackfridays), 'Black Friday', df['specialday'])
    df['specialday'] = np.where(df['date'].isin(cybermondays), 'Cyber Monday', df['specialday'])
    df['specialday'] = np.where(df['date'].isin(blackweekend), 'Black Weekend', df['specialday'])
    
    # easter saturdays
    eastersundays = df[df['holiday']=='Påskedag']['date'].tolist()
    easterdays = [d + relativedelta(days=1) for d in eastersundays]
    df['specialday'] = np.where(df['date'].isin(easterdays), 'Påskeåbent', df['specialday'])
    
    # christmas
    christmaseves = df[(df.date.dt.month==12) & (df.date.dt.day==24)]['date'].tolist()
    christmasdays = [pd.date_range(d, d.replace(day=31)) for d in christmaseves]
    christmasdays = [d for sublist in christmasdays for d in sublist]
    cond = df.date.isin(christmasdays) & (df['is-holiday']==0)
    df['specialday'] = np.where(cond, 'Juleåbent', df['specialday'])
    
    # distance until christmas since black friday
    # black fridays
    bf = df[df['specialday']=='Black Friday'].groupby('year')['date'].min().reset_index()
    bf.rename({'date':'bf'}, inplace=True, axis=1)
    df = pd.merge(df, bf, on=['year'], how='left')
    
    # christmas
    cm = df[(df.date.dt.month==12) & (df.date.dt.day==24)].groupby('year')['date'].min().reset_index()
    cm.rename({'date':'cm'}, axis=1, inplace=True)
    df = pd.merge(df, cm, on=['year'], how='left')
    
    # distance
    df['until-christmas'] = np.where((df['date']>=df['bf']) & (df['date']<=df['cm']), (df['date']-df['bf']).dt.days/(df['cm'] - df['bf']).dt.days, 0)
    
    # drop helper columns
    df.drop(['cm', 'bf'], axis=1, inplace=True)

    return df


def add_financial_year(df, fy_start_month):
    df['fy'] = np.where(df['month']<fy_start_month, df['year'] % 1000 - 1, df['year'] % 1000)
    df['fy'] = df['fy'].astype(str) + (df['fy']+1).astype(str)
    df['fy'] = df['fy'].astype(int)
    return df


def create_calendar(startdate, enddate, fy_start_month=4):
    # add padding to dates
    startdate = startdate-relativedelta(months=13)
    enddate = enddate+relativedelta(months=13)
    # get standard features
    df = create_dates_df(startdate, enddate)
    df = add_standard_date_columns(df)
    df = add_financial_year(df, fy_start_month)
    df = add_holiday_cols(df)
    df = add_payday_columns(df)
    df = add_specialdays(df, startdate, enddate)
    df = df.loc[(df['date'].dt.date >= startdate) & (df['date'].dt.date <= enddate)]
    return df

def encode_cyclic(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * (df[col]-1)/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * (df[col]-1)/max_val)
    return df

def Get_Date_Features(startdate, enddate):
    # add padding to dates
    startdate = startdate-relativedelta(months=13)
    enddate = enddate+relativedelta(months=13)
    # get standard features
    df = create_dates_df(startdate, enddate)
    df = add_standard_date_columns(df)
    # add holidays
    df = add_holiday_cols(df)
    # df['is-holiday'] = np.where(df['holiday']!="", 1, 0)
    # add payday features
    df = add_payday_columns(df)
    # add special days
    df = add_specialdays(df, startdate, enddate)

    # encode other categorical features
    to_encode = ['month', 'weeknum', 'weekday', 'holiday', 'paymonth', 'specialday']
    for col in to_encode:
        # Get one hot encoding of column
        one_hot = pd.get_dummies(df[col])
        one_hot = one_hot.add_prefix(col + '_')
        # Drop column as it is now encoded
        df = df.drop(col,axis = 1)
        # Join the encoded df
        df = df.join(one_hot)
        # drop empty encoded column
        try:
            df.drop([col+'_'], axis=1, inplace=True)
        except:
            pass
        
    # drop unnecessary features
    df.drop(['quarter', 'day'], axis=1, inplace=True)

    return df


if __name__ == "__main__":
    start = date(2022,3,10) #'2022-03-10'
    end = date(2023,1,3) #'2022-12-01'
    df = Get_Date_Features(start, end)
    print(df.head())
