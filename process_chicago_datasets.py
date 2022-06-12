import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import glob
import geopandas as gp
plt.rcParams['figure.figsize'] = (12,9)
plt.rcParams['figure.facecolor']='w'
import warnings
warnings.filterwarnings('ignore')


def get_area_n(fname):
    """
    returns community area number
    """

    comm_n = fname.split('_')[2]

    return int(comm_n)


get_comm_area = lambda c: int(c.split('_')[1])

get_census_tract = lambda n: int(str(n)[:-4])

# IMPORT CHICAGO SHAPEFILE

chicago_shp = '../data_misc/chicago_shapefile/geo_export_ec346dbb-8c11-44b4-be04-0301ae3f9c64.shp'
chicago = gp.read_file(chicago_shp)
cols = ['countyfp10','namelsad10', 'name10','notes','statefp10','tractce10']
chicago.drop(columns=cols, inplace=True)
chicago.geoid10 = chicago.geoid10.apply(np.int64)

###############################################################################
###############################################################################
###################### Processing Chicago Temperature Data ####################
###############################################################################
###############################################################################
path = 'chicago_nsrdb/*.csv'
files = glob.glob(path)
files.sort()
N = len(files)

heatwave_threshold = 32 # degrees celcius

print('Concatenating temperature data... ')

frames = []
for f in files:
    n = get_area_n(f)
    df = pd.read_csv(f,
                     usecols=['time',
                              f'Temp_{n}',
                              ],
                     index_col='time',
                     parse_dates=True
                     )
    frames.append(df)
full_df = pd.concat(frames, axis=1)
heatwaves = full_df.copy()
full_df['average'] = full_df.mean(axis=1)
hot_times = full_df[full_df.average > heatwave_threshold]

print('Calculating temperature anomaly during heatwaves. \n')
print('***The temperature anomaly is defined as the difference between')
print('community area average temperature and citywide average temperature.***\n')

for col in hot_times.columns:
    if col =='average':
        continue
    else:
        new_col = col+'_A'
        hot_times[new_col] = (hot_times[col] - hot_times['average'])

avg_ta = pd.DataFrame(hot_times.iloc[:,N+1:].mean(axis=0))
avg_ta['commarea_n'] = avg_ta.index
avg_ta['commarea_n'] = avg_ta['commarea_n'].apply(get_comm_area)
avg_ta.reset_index(drop=True, inplace=True)
avg_ta.rename(columns={0:'H_a'}, inplace=True)
avg_ta['H_amin'] = avg_ta['H_a'] - avg_ta['H_a'].min()

print('Merging chicago shapefile and temperature anomaly.')
len_before = len(chicago)
n_cols_before = len(chicago.columns)
chicago = pd.merge(chicago, avg_ta, on=['commarea_n'])
len_after = len(chicago)
n_cols_after = len(chicago.columns)

print(f'Expecting {len_before} rows, got {len_after} rows.')
print(f'Expecting {n_cols_before+len(avg_ta.columns)} columns {n_cols_after}\n')

###############################################################################
###############################################################################
###################### Processing Chicago Population Data #####################
###############################################################################
###############################################################################
print('Opening 2010 Census Data...')
pop_df = pd.read_csv('Population_by_2010_Census_Block.csv')
print('Getting census tract number from census block (downscaling)...')
pop_df['geoid10'] = pop_df['CENSUS BLOCK FULL'].apply(get_census_tract)
print('Summing the population in each census tract...')
pop_agg = pop_df[['TOTAL POPULATION','geoid10']].groupby('geoid10').sum()

print('Merging Chicago shapefile and population data')
len_before = len(chicago)
n_cols_before = len(chicago.columns)
chicago = chicago.merge(pop_agg, on='geoid10',how='left')
len_after = len(chicago)
n_cols_after = len(chicago.columns)

print(f'Expecting {len_before} rows, got {len_after} rows.')
print(f'Expecting {n_cols_before+len(pop_agg.columns)} columns {n_cols_after}\n')
###############################################################################
###############################################################################
######################### Processing Chicago Crime Data #######################
###############################################################################
###############################################################################

crimes_of_interest = ['ARSON', 'ASSAULT', 'BATTERY', 'BURGLARY',
                      'CONCEALED CARRY LICENSE VIOLATION', 'CRIMINAL DAMAGE',
                      'CRIMINAL SEXUAL ASSAULT', 'CRIMINAL TRESPASS',
                      'HOMICIDE', 'HUMAN TRAFFICKING',
                      'INTIMIDATION', 'KIDNAPPING',
                      'MOTOR VEHICLE THEFT', 'NARCOTICS',
                      'OBSCENITY', 'OFFENSE INVOLVING CHILDREN',
                      'OTHER NARCOTIC VIOLATION', 'PROSTITUTION',
                      'PUBLIC INDECENCY', 'PUBLIC PEACE VIOLATION', 'ROBBERY',
                      'SEX OFFENSE', 'STALKING', 'THEFT', 'WEAPONS VIOLATION']

violent_crimes = ['ASSAULT', 'BATTERY', 'CRIMINAL SEXUAL ASSAULT','HOMICIDE',
                  'ROBBERY']

print('Opening Chicago Crime Data from 2021')
crime_df = pd.read_csv('Crimes_-_Map.csv', usecols=[' PRIMARY DESCRIPTION',
                                                    'WARD',
                                                    'LATITUDE',
                                                    'LONGITUDE',
                                                    'DATE  OF OCCURRENCE'],
                       parse_dates=True,
                       index_col='DATE  OF OCCURRENCE')
crime_df.dropna(inplace=True)
crime_df = gp.GeoDataFrame(
    crime_df, geometry=gp.points_from_xy(crime_df.LONGITUDE, crime_df.LATITUDE)
)

coi = lambda x: x in crimes_of_interest
violent = lambda x: x in violent_crimes

print('Counting crimes in each census tract')
crime_df['to_count'] = crime_df[' PRIMARY DESCRIPTION'].apply(coi)
crime_df['is_violent'] = crime_df[' PRIMARY DESCRIPTION'].apply(violent)
crime_df = crime_df[crime_df.to_count == True]

dfsjoin = gp.sjoin(chicago, crime_df)
n_crimes = dfsjoin[['geoid10','to_count']].groupby('geoid10').sum()
n_violent = dfsjoin[['geoid10','is_violent']].groupby('geoid10').sum()
new_crime_df = n_crimes.merge(n_violent, on='geoid10')


print('Merging Chicago shapefile and crime data')
len_before = len(chicago)
n_cols_before = len(chicago.columns)
chicago = chicago.merge(new_crime_df, on='geoid10')
chicago.rename(columns={'to_count':'crime_count'},inplace=True)
len_after = len(chicago)
n_cols_after = len(chicago.columns)

print(f'Expecting {len_before} rows, got {len_after} rows.')
print(f'Expecting {n_cols_before+len(new_crime_df.columns)} columns {n_cols_after}\n')

# chicago['crime_rate'] = np.divide(chicago.crime_count,
#                                   chicago['TOTAL POPULATION'])

###############################################################################
###############################################################################
########################## Processing Chicago Park Data #######################
###############################################################################
###############################################################################

print('Reading in Building Location Data')
churches = gp.read_file('kx-chicago-illinois-churches-SHP/chicago-illinois-churches.shp')
parks = gp.read_file('kx-chicago-illinois-parks-SHP/chicago-illinois-parks.shp')
schools = gp.read_file('kx-chicago-illinois-schools-SHP/chicago-illinois-schools.shp')
libraries = gp.read_file('kx-chicago-illinois-libraries-SHP/chicago-illinois-libraries.shp')

churches.to_crs({'init': 'epsg:4326'},inplace=True)
parks.to_crs({'init': 'epsg:4326'},inplace=True)
libraries.to_crs({'init': 'epsg:4326'},inplace=True)
schools.to_crs({'init': 'epsg:4326'},inplace=True)

private = schools[schools.TYPE=='PRIVATE']
public = schools[schools.TYPE=='CPS']

churches['n_churches'] = 1
libraries['n_libraries'] = 1
private['n_private'] = 1
public['n_public'] = 1

print('Calculating census tract area... ')
chi_copy = chicago.copy()
chi_copy = chi_copy.to_crs(epsg=6933)
chi_copy['area'] = chi_copy.geometry.area/1e6
chicago['tract_area'] = chi_copy['area']

parkjoin = gp.overlay(chicago, parks[['geometry']], how='intersection')

print('Calculating park area... ')
parks_copy = parkjoin.copy()
parks_copy = parks_copy.to_crs(epsg=6933)
parks_copy['park_area'] = parks_copy.geometry.area/1e6
parkjoin['park_area'] = parks_copy['park_area']

pct_park = parkjoin[['geoid10','tract_area', 'park_area']]
pct_park.geoid10 = pct_park.geoid10.apply(np.int64)
pct_park['pct_park'] = np.divide(pct_park['park_area'],pct_park['tract_area'])
pct_park = pct_park.groupby('geoid10').sum().reset_index()
# pct_park.set_index('geoid10', inplace=True)

# merging chicago and park area
print('Merging Chicago shapefile and park area')
len_before = len(chicago)
n_cols_before = len(chicago.columns)
chicago = chicago.merge(pct_park[['pct_park', 'geoid10']], on='geoid10', how='left')
chicago['pct_park'].fillna(0, inplace=True)
len_after = len(chicago)
n_cols_after = len(chicago.columns)
print(f'Expecting {len_before} rows, got {len_after} rows.')
print(f'Expecting {n_cols_before+1} columns {n_cols_after}\n')

publicjoin = gp.sjoin(chicago, public)
privatejoin = gp.sjoin(chicago, private)
librariesjoin = gp.sjoin(chicago, libraries)
churchesjoin = gp.sjoin(chicago, churches)

###############################################################################
###############################################################################
######################## Processing Chicago Building Data #####################
###############################################################################
###############################################################################


print('counting the number of buildings in each census tract... ')
# # using groupby sum on geoid10 makes geoid10 the default index
n_cps = publicjoin[['geoid10','n_public']].groupby('geoid10').sum()
n_private = privatejoin[['geoid10','n_private']].groupby('geoid10').sum()
n_libraries = librariesjoin[['geoid10','n_libraries']].groupby('geoid10').sum()
n_churches = churchesjoin[['geoid10','n_churches']].groupby('geoid10').sum()
chicago_buildings = pd.concat([n_churches, n_cps, n_libraries, n_private], axis=1)
chicago_buildings.fillna(0,inplace=True)

print('Merging Chicago shapefile and building locations')
len_before = len(chicago)
n_cols_before = len(chicago.columns)
chicago.set_index('geoid10',inplace=True)
chicago = chicago.merge(chicago_buildings[['n_churches',
                                           'n_public',
                                           'n_private',
                                           'n_libraries']],
                                           on='geoid10',
                                           how='left')
chicago.reset_index(inplace=True)
len_after = len(chicago)
n_cols_after = len(chicago.columns)

print(f'Expecting {len_before} rows, got {len_after} rows.')
print(f'Expecting {n_cols_before+4} columns {n_cols_after}\n')

###############################################################################
###############################################################################
######################## Processing Google Sunroof Data #######################
###############################################################################
###############################################################################

print('Adding Google Sunroof data')
google_sunroof = 'project-sunroof-census_tract.csv'
google_df = pd.read_csv(google_sunroof,
                        usecols=['region_name',
                                 'percent_qualified',
                                 'number_of_panels_total',
                                 'kw_total',
                                 'existing_installs_count'])
google_df.rename(columns={'region_name':'geoid10'}, inplace=True)

print('Merging Chicago shapefile and park area')
len_before = len(chicago)
n_cols_before = len(chicago.columns)
chicago = pd.merge(chicago, google_df, on=['geoid10'], how='left')
len_after = len(chicago)
n_cols_after = len(chicago.columns)
print(f'Expecting {len_before} rows, got {len_after} rows.')
print(f'Expecting {n_cols_before+4} columns {n_cols_after}\n')


###############################################################################
###############################################################################
######################### Processing Socioeconomic Data #######################
###############################################################################
###############################################################################

print('Reading Chicago Socioeconomic Data...')
econ_data = 'Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012.csv'
chicago_econ = pd.read_csv(econ_data)
chicago_econ.rename(columns={'Community Area Number':'commarea_n'}, inplace=True)


print('Merging Chicago shapefile and park area')
len_before = len(chicago)
n_cols_before = len(chicago.columns)
chicago = pd.merge(chicago, chicago_econ[['commarea_n',
                                          'COMMUNITY AREA NAME',
                                          'PERCENT OF HOUSING CROWDED',
                                          'PERCENT HOUSEHOLDS BELOW POVERTY',
                                          'PERCENT AGED UNDER 18 OR OVER 64',
                                          'PER CAPITA INCOME ',
                                          'HARDSHIP INDEX']],
                   on=['commarea_n'], how='left')
len_after = len(chicago)
n_cols_after = len(chicago.columns)
print(f'Expecting {len_before} rows, got {len_after} rows.')
print(f'Expecting {n_cols_before+8} columns {n_cols_after}\n')

print(list(chicago.columns))

chicago.to_file('processed_data/chicago_data.shp')

if __name__ == "__main__":

    import matplotlib.colors as colors
    fig, ax = plt.subplots(3,2,figsize=(14,12))
    norm = colors.Normalize(vmin=-2, vmax=2)
    cmap = 'coolwarm'
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    chicago.plot(ax=ax[0][0], edgecolor='k',
                 linewidth=0.25,column='H_a',
                 norm=norm, cmap=cmap, legend=False)
    ax_cbar = fig.colorbar(cbar, ax=ax[0][0], shrink=0.6)
    ax_cbar.set_label(r'H$_a$ [$\Delta^\circ$C]')
    ax[0][0].set_title('Chicago Heatwave Temperature Anomaly \n Community Area')
    ax[0][0].set_axis_off()

    norm = colors.Normalize(vmin=0, vmax=chicago['TOTAL POPULATION'].max())
    cmap = 'cividis'
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    chicago.plot(ax=ax[0][1], edgecolor='k',
                 linewidth=0.25,column='TOTAL POPULATION',
                 norm=norm, cmap=cmap, legend=False)
    ax_cbar = fig.colorbar(cbar, ax=ax[0][1], shrink=0.6)
    ax_cbar.set_label(r'Number of People')
    ax[0][1].set_title('Chicago Population from the 2010 Census')
    ax[0][1].set_axis_off()

    norm = colors.Normalize(vmin=0, vmax=1000)
    cmap = 'plasma'
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    chicago.plot(ax=ax[1][0], edgecolor='k',
                 linewidth=0.25,column='crime_count',
                 norm=norm, cmap=cmap, legend=False)
    ax_cbar = fig.colorbar(cbar, ax=ax[1][0], shrink=0.6)
    ax_cbar.set_label(r'Number of Crimes Reported')
    ax[1][0].set_title('Chicago Crimes from 2021')
    ax[1][0].set_axis_off()

    norm = colors.Normalize(vmin=0, vmax=0.8)
    cmap = 'viridis'
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    chicago.plot(ax=ax[1][1], edgecolor='k',
                 linewidth=0.25,column='pct_park',
                 norm=norm, cmap=cmap, legend=False)
    ax_cbar = fig.colorbar(cbar, ax=ax[1][1], shrink=0.6)
    ax_cbar.set_label(r'Percent Area Covered by Park')
    ax[1][1].set_title('Chicago Park Area\n Census Tract')
    ax[1][1].set_axis_off()

    norm = colors.Normalize(vmin=0, vmax=100)
    cmap = 'inferno'
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    chicago.plot(ax=ax[2][0], edgecolor='k',
                 linewidth=0.25,column='percent_qualified',
                 norm=norm, cmap=cmap, legend=False)
    ax_cbar = fig.colorbar(cbar, ax=ax[2][0], shrink=0.6)
    ax_cbar.set_label('Percent Solar-Qualified Rooftops')
    ax[2][0].set_title('Chicago Rooftop Solar')
    ax[2][0].set_axis_off()

    norm = colors.Normalize(vmin=0, vmax=100)
    cmap = 'magma'
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    chicago.plot(ax=ax[2][1], edgecolor='k',
                 linewidth=0.25,column='HARDSHIP INDEX',
                 norm=norm, cmap=cmap, legend=False)
    ax_cbar = fig.colorbar(cbar, ax=ax[2][1], shrink=0.6)
    ax_cbar.set_label('Hardship Index')
    ax[2][1].set_title('Chicago Socioeconomic Data')
    ax[2][1].set_axis_off()
    plt.tight_layout()
    plt.show()
