# -*- coding: utf-8 -*-

import pandas as pd

def create_dates_df(startdate, enddate):
    dates = pd.date_range(start=startdate, end=enddate)
    df = pd.DataFrame(dates)
    return df

