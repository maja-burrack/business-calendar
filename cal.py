
import pandas as pd
import numpy as np
import holidays

start = '2022-01-01'
end = '2022-05-01'

def create_dates_df(startdate, enddate):
    dates = pd.date_range(start=startdate, end=enddate)
    df = pd.DataFrame(dates)
    df.rename({0: 'date'}, axis=1, inplace=True)
    return df

def add_columns(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weeknum'] = df['date'].dt.isocalendar().week
    return df

def add_holidays(df):
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
    
    return df

