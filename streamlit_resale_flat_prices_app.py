# conda deactivate
# conda env remove -n streamlit_resale_flat_prices
# conda create -n streamlit_resale_flat_prices python=3.8.5
# conda activate streamlit_resale_flat_prices
# conda install streamlit
# cd C:\Users\Russ\Anaconda3\Russ Projects\Resale Flat Prices\streamlit_resale_flat_prices
# python streamlit_resale_flat_prices_app.py


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import streamlit as st

# title of app
st.title('Resale Flat Prices')

# cache data for quicker loading
@st.cache
# load data from csv
def load_data_from_csv(file_path):
    # read data
    data_raw = pd.read_csv(file_path)
    # convert to year_month to datetime format
    data_raw['year_month'] = pd.to_datetime(data_raw['year_month'])
    # copy raw data
    data = data_raw.copy()

    # return 
    return data

# load data from csv
data = load_data_from_csv('resale_flat_prices_clean_data.csv')




# slider to select past n number of years of data to use

# set default value to past 10 years
# set to use at least past 1 year of data
# determine max number of years from data
max_past_n_years = round((data['year_month'].max() - data['year_month'].min()) / np.timedelta64(1, 'Y')) + 1
# define slider
past_n_years = st.slider('How many years of past data would you like to use?', min_value=1, max_value=max_past_n_years, value=10)

# filter number of years of data to use based on slider

# define latest year of data
latest_year = str(data['year_month'].max().year - past_n_years)
# filter past n number of years
data = data.loc[(data['year_month'] >= latest_year+'-01-01')]



### map using latitude and longitude ###
# map
st.map(data)



### boxplot of flat type ###

# set plot and figure size
fig, ax = plt.subplots(figsize=(15,10))

# order by flat_type alphabetically
flat_type_order = sorted(list(data['flat_type'].unique()))
# plot ax
ax = sns.boxplot(
    x='resale_price', 
    y='flat_type', 
    data=data,
    order=flat_type_order, 
    flierprops={'marker':'.', 'alpha':0.05}
    )

# formatting
# add thousands separator
ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
# set title
ax.set_title('Boxplot of Flat Resale Price by Town')

# show plot
st.pyplot(fig)


### boxplot of town ###

# set plot and figure size
fig, ax = plt.subplots(figsize=(15,10))

# order by descending median resale_price
town_order = list(data.groupby(['town']).agg({'resale_price':'median'}).reset_index().sort_values('resale_price', ascending=False)['town'])
# plot boxplot
sns.boxplot(
    x='resale_price', 
    y='town', 
    data=data, 
    order=town_order,
    flierprops={'marker':'.', 'alpha':0.05}
    )

# formatting
# add thousands separator
ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
# set title
ax.set_title('Boxplot of Flat Resale Price by Town')
# show ploy
st.pyplot(fig)


print('### File ran properly ###')