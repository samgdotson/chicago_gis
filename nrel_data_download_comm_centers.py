
import numpy as np
import sys, os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from nrel_api import *


coords = pd.read_csv('commarea_centers.csv', usecols=['commarea', 'longitude','latitude'])

plant_names = list(coords.commarea)
lons = list(coords.longitude)
lats = list(coords.latitude)

parameters['attr_list'] = ['air_temperature',
                           'relative_humidity',
                           'ghi',
                           'wind_speed',
                           'surface_pressure',
                           'surface_albedo']


last_idx = np.where(np.array(plant_names)==66)[0][0]

years = np.arange(2000,2021,1).astype('int')
full_frames = []
k = 0
N = len(plant_names[last_idx:])
# N = len(plant_names[last_idx:])
# for n, i, j in zip(plant_names[last_idx:], lats[last_idx:],lons[last_idx:]):
for n, i, j in zip(plant_names[last_idx:], lats[last_idx:],lons[last_idx:]):
    print(f" ({k}/{N*len(years)}) Getting data for coordinates: {i}, {j} -- {n}")
    parameters['lon'] = j
    parameters['lat'] = i
    frames = []
    # get data for several years from this location
    if (k + len(years)) > 480:
        print('Download limit reached.')
        break
    for m,year in enumerate(years):
        print(f" ({k}/{N*len(years)}) -- {year}")
        parameters['year'] = year
        URL = make_csv_url(parameters=parameters, personal_data=personal_data, kind='solar')
#         print(URL)
        df = pd.read_csv(URL, skiprows=2)
        cols=['Year','Month', 'Day', 'Hour', 'Minute']
        df['time'] = pd.to_datetime(df[cols])
        df.drop(columns=cols, inplace=True)
        df.set_index('time', inplace=True)
        df.rename(columns={'Temperature':f'Temp_{n}',
                           'Wind Speed':f'Wind_{n}',
                           'Relative Humidity':f'RH_{n}',
                           'Surface Albedo':f'SA_{n}',
                           'Pressure':f'P_{n}',
                           'GHI':f'GHI_{n}'}, inplace=True)
        frames.append(df)
        k += 1
    full_df = pd.concat(frames, axis=0)
    full_df.to_csv(f'chicago_nsrdb/commarea_{n}_weather_2000_2020.csv')
    # full_frames.append(full_df)
# total_t = pd.concat(full_frames, axis=1)
