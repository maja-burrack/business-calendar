# -*- coding: utf-8 -*-

import pandas as pd

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

