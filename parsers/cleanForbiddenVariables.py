import pandas as pd
import numpy as np

df = pd.read_csv('../datasets/2008.csv')

df.drop('ArrTime', axis=1, inplace=True)
df.drop('ActualElapsedTime', axis=1, inplace=True)
df.drop('AirTime', axis=1, inplace=True)
df.drop('TaxiIn', axis=1, inplace=True)
df.drop('Diverted', axis=1, inplace=True)
df.drop('CarrierDelay', axis=1, inplace=True)
df.drop('WeatherDelay', axis=1, inplace=True)
df.drop('NASDelay', axis=1, inplace=True)
df.drop('SecurityDelay', axis=1, inplace=True)
df.drop('LateAircraftDelay', axis=1, inplace=True)

df.to_csv('../datasets/processed/2008.csv')