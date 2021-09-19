# conda deactivate
# conda env remove -n streamlit_resale_flat_prices
# conda create -n streamlit_resale_flat_prices python=3.8.5
# conda activate streamlit_resale_flat_prices
# conda install streamlit
# python streamlit_resale_flat_prices_app.py

# imports
import pandas as pd
import numpy as np
import streamlit as st
import os
import datetime as dt
from dateutil.relativedelta import relativedelta
import requests
import pickle

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pydeck as pdk



# title of app
st.title('Exploration Resale Prices of Public Housing in Singapore')
st.write('\n')
st.write('\n')



### load data ###

# cache data for quicker loading
@st.cache
# load data from csv
def load_data_from_csv(file_path):
    # read data
    data = pd.read_csv(file_path)
    # convert to year_month to datetime format
    data['year_month'] = pd.to_datetime(data['year_month'])
    # convert latitude and longitude to numeric
    data[['latitude', 'longitude']] = data[['latitude', 'longitude']].apply(pd.to_numeric)
    # return 
    return data

# define data directory
data_folder = 'data'
# create empty dataframe to concat data later
data = pd.DataFrame()
# loop through each file in data_folder and concat into one df
for file_name in os.listdir(data_folder):
    # load data
    temp_data = load_data_from_csv(os.path.join(data_folder, file_name))
    # concat data
    data = pd.concat([data, temp_data])

# get date range of full data set
min_data_date = data['year_month'].min().strftime('%B %Y')
max_data_date = data['year_month'].max().strftime('%B %Y')
total_row_count = '{:,}'.format(len(data))

# introduction to app
st.write(f'The data has been extracted from Data.gov.sg.')
st.write(f'There are a total of {total_row_count} recorded resale flat transactions from {min_data_date} to {max_data_date}.')
st.write('\n')
st.write('\n')



### map using latitude and longitude ###

# cache data
@st.cache
# filter data for map due to memory limit in streamlit
def filter_data_for_map(data, years):
    # get number of years to filter
    map_date = data['year_month'].max() - relativedelta(years=years)
    # filter data for map by years
    data_for_map = data.loc[(data['year_month'] >= map_date)]
    # return
    return data_for_map

# filter data for map due to memory limit in streamlit
# define max number of years
max_heatmap_years = 1
data_for_map = filter_data_for_map(data, max_heatmap_years)

# describe map
st.write(f'Let\'s have a look at where flats have been transacted in the past {max_heatmap_years} year. The taller the pillars, the more transactions have taken place.')
st.write('The spikes are likely to be flats where the [Minimum Occupation Period](https://www.hdb.gov.sg/residential/selling-a-flat/eligibility) has just passed and are eligible to be sold on the resale market.')

# create pydeck map in streamlit
def pydeck_map(data_for_map):
    # create map from pydeck
    layer=[
        pdk.Layer(
            'HexagonLayer',
            data=data_for_map,
            get_position='[longitude, latitude]',
            radius=40,
            elevation_scale=3,
            elevation_range=[0,500],
            pickable=True,
            extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=data_for_map,
            get_position='[longitude, latitude]',
            get_color='[200, 30, 0, 160]',
            get_radius=40,
        )
    ]

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(latitude=1.355, longitude=103.81, zoom=11, pitch=40),
        layers=layer
        ))

# create pydeck map in streamlit
pydeck_map(data_for_map)
st.write('\n')
st.write('\n')


### silder to fitler data period ###

# slider to select period of years to visualise

# describe visualisation section
st.write(f'Next up is a deeper dive into the Resale Flat Prices data.')
# default minimum and maximum year from data
period_date_max = data['year_month'].max().year
period_date_min = data['year_month'].min().year
# define slider, default value max as latest data year and min as x years before
# define years before
default_years_before = 3
visualisation_period = st.slider('Select a period would you like to visualise.', min_value=period_date_min, max_value=period_date_max, value=(period_date_max-default_years_before,period_date_max))
# filter data based on period
data = data.loc[(data['year_month'] >= str(visualisation_period[0])+'-01-01') & (data['year_month'] <= str(visualisation_period[1])+'-12-01')]

# get descriptive statistics of resale_price for selected period

# describe resale_price
period_describe = data['resale_price'].describe()
# mean of resale price formatted
period_mean = '{:,}'.format(round(period_describe['mean']))
# median of resale price formatted
period_median = '{:,}'.format(round(period_describe['50%']))
# max resale price formatted
period_max = '{:,}'.format(round(period_describe['max']))
# get data for most expensive flat, if there is more than one, select latest transaction
most_expensive = data.loc[(data['resale_price'] == period_describe['max'])].reset_index(drop=True).sort_values('year_month', ascending=False).iloc[0]

# print descriptive statistics
st.write(f'In the period from {visualisation_period[0]} to {visualisation_period[1]}:')
st.write(f'The average resale flat price is ${period_mean} and the median is ${period_median}.')
st.write(f"The most expensive flat sold for ${period_max}! The flat was transacted in {most_expensive['year_month'].strftime('%B %Y')} at {most_expensive['block'].title()+' '+most_expensive['street_name'].title()}, it was a {most_expensive['flat_type'].title()} with a {round(most_expensive['floor_area_sqm'])} square meter floor area.")
st.write('\n')
st.write('\n')



### visualise data ###

# set plot attributes
plot_figsize = (15,10)
plot_title_fontsize = 18
plot_axis_fontsize = 15



### histogram of town ###

# order by value count of town
town_order = data['town'].value_counts().index
# describe plot
st.write(f"This chart shows the number of transactions by Town. The most transactions occured in {town_order[0].title()}.")

# set plot and figure size
fig, ax = plt.subplots(figsize=plot_figsize)

# plot ax
ax = sns.countplot(
    y='town',
    data=data,
    order=town_order
)

# formatting
# add thousands separator
ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
# set title
ax.set_title('Count of Transactions by Town', fontsize=plot_title_fontsize)
ax.set_xlabel('Count', fontsize=plot_axis_fontsize)
ax.set_ylabel('Town', fontsize=plot_axis_fontsize)

# show plot
st.pyplot(fig)
st.write('\n')



### histogram of flat type ###

# describe plot
st.write(f"And this chart shows the number of transactions by Flat Type. The most often transacted Flat Type is {data['flat_type'].value_counts().index[0].title()}")

# set plot and figure size
fig, ax = plt.subplots(figsize=plot_figsize)

# order by flat_type alphabetically
flat_type_order = sorted(list(data['flat_type'].unique()))

# plot ax
ax = sns.countplot(
    x='flat_type',
    data=data,
    order=flat_type_order
)

# formatting
# add thousands separator
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
# set title
ax.set_title('Count of Transactions by Flat Type', fontsize=plot_title_fontsize)
ax.set_xlabel('Flat Type', fontsize=plot_axis_fontsize)
ax.set_ylabel('Count', fontsize=plot_axis_fontsize)

# show plot
st.pyplot(fig)
st.write('\n')



### histogram of flat prices ###

# describe plot
st.write(f"Here's a chart that shows the distribution of Resale Prices by Flat Type.")

# set plot and figure size
fig, ax = plt.subplots(figsize=plot_figsize)

# plot ax
ax = sns.histplot(
    x='resale_price',
    data=data.sort_values('flat_type'),
    bins=100,
    hue='flat_type',
    element='step',
    kde=True
)

# formatting
# add thousands separator
ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
# set title
ax.set_title('Resale Flat Price by Flat Type', fontsize=plot_title_fontsize)
ax.set_xlabel('Resale Price', fontsize=plot_axis_fontsize)
ax.set_ylabel('Count', fontsize=plot_axis_fontsize)

# show plot
st.pyplot(fig)
st.write('\n')



### scatterplot of resale_price and floor_area_sqm ###

# describe plot
st.write(f"Now this is a scatterplot of Resale Prices compared to Floor Area. No surprise here that we can see a general trend that a flat with more rooms and space cost more.")
st.write(f"However, we can also see that similar flats with the same floor area and flat type can vary significantly in price!")

# set plot and figure size
fig, ax = plt.subplots(figsize=plot_figsize)

# plot ax
sns.scatterplot(
    x='floor_area_sqm',
    y='resale_price',
    data=data.sort_values('flat_type'),
    hue='flat_type',
    alpha=0.4
)

# formatting
# add thousands separator
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
# set title
ax.set_title('Resale Flat Price vs Floor Area', fontsize=plot_title_fontsize)
ax.set_xlabel('Floor Area (SQM)', fontsize=plot_axis_fontsize)
ax.set_ylabel('Resale Price', fontsize=plot_axis_fontsize)

# show plot
st.pyplot(fig)
st.write('\n')



### boxplot of flat type ###

# describe plot
st.write("These are boxplots of Resale Prices by different Flat Types. Here's a quick refresher on how to [read boxplots](https://miro.medium.com/max/2400/1*2c21SkzJMf3frPXPAR_gZA.png) if you need it.")

# set plot and figure size
fig, ax = plt.subplots(figsize=plot_figsize)

# order by flat_type alphabetically
flat_type_order = sorted(list(data['flat_type'].unique()))

# plot ax
ax = sns.boxplot(
    x='flat_type', 
    y='resale_price', 
    data=data,
    order=flat_type_order, 
    flierprops={'marker':'.', 'alpha':0.05}
    )

# formatting
# add thousands separator
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
# set title
ax.set_title('Resale Flat Price by Flat Type', fontsize=plot_title_fontsize)
ax.set_xlabel('Flat Type', fontsize=plot_axis_fontsize)
ax.set_ylabel('Resale Price', fontsize=plot_axis_fontsize)

# show plot
st.pyplot(fig)
st.write('\n')



### boxplot of town ###

# group by town and get median resale price
town_median_resale_price = data.groupby(['town']).agg({'resale_price':'median'}).reset_index().sort_values('resale_price', ascending=False)
# get town with highest median 
most_expensive_town = town_median_resale_price['town'].iloc[0].title()
# get town with lowest median
least_expensive_town = town_median_resale_price['town'].iloc[-1].title()
# compare and get difference between highest median and lowest median
median_price_difference = round(town_median_resale_price['resale_price'].iloc[0] / town_median_resale_price['resale_price'].iloc[-1], 1)
# order by descending median resale_price
town_order = list(town_median_resale_price['town'])

# describe plot
st.write(f"Here are more pretty boxplots of Resale Prices by Town. The most expensive area to buy a flat is {most_expensive_town}.")
st.write(f"Apart from the floor area and flat type, which town the flat is in also influences the price. The median price of a flat in {most_expensive_town} is {median_price_difference} times higher than {least_expensive_town}!")

# set plot and figure size
fig, ax = plt.subplots(figsize=plot_figsize)

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
# set title and axis
ax.set_title('Resale Flat Price by Town', fontsize=plot_title_fontsize)
ax.set_xlabel('Resale Price', fontsize=plot_axis_fontsize)
ax.set_ylabel('Town', fontsize=plot_axis_fontsize)

# show ploy
st.pyplot(fig)
st.write('\n')


### prediction ###

# prediction section

st.write('\n')
st.write('# Predict Resale Flat Price')
st.write('Enter some basic information (or just try it out with the default values) of the flat you want to sell or buy for the model to predict it\'s price.')

# form to store users input
with st.form(key='input_form'):

    # ask and store users input
    input_postal_code = st.text_input(label='Postal Code', value='440033')
    input_floor_area_sqm = st.number_input(label='Floor Area (square meters)', min_value=1, max_value=500, value=70, step=10)
    input_floor = st.number_input(label='Floor', min_value=1, max_value=100, value=10, step=2)
    input_lease_commence_year = st.number_input(label='Lease Commence (year)', min_value=1900, max_value=dt.date.today().year, value=1975, step=1)

    # get coordinates from address as latitude and longitude using google geocode api
    def get_coordinates_from_address(address, api_key):
        '''
        get coodinates from an address using google geocode api
        information on how to set up and create api key can be found here
        https://developers.google.com/maps/documentation/geocoding/overview?hl=en_GB

        arguments:
        address (str): address to get coordinates of
        api_key (str): api key from google cloud platform

        returns:
        a tuple containing latitude and longitude
        '''
        # request response from google geocode api
        api_response = requests.get(f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}').json()
        # check if api response is 'OK'
        if api_response['status'] == 'OK':
            # get latitude from response
            latitude = api_response['results'][0]['geometry']['location']['lat']
            # get longitude from response
            longitude = api_response['results'][0]['geometry']['location']['lng']
        else:
            # if status is not 'OK', add status as error message
            latitude = 'error: ' + api_response['status']
            longitude = 'error: ' + api_response['status']

        # return a tuple
        return (latitude, longitude)

    # get latitude and longitude from postal code
    coordinates = get_coordinates_from_address(input_postal_code+' Singapore', st.secrets['geocode_api_key'])
    # calculate remaining lease years from lease commencement date
    input_remaining_lease_years = dt.date.today().year - input_lease_commence_year

    # format user inputs into df for xgb prediction
    input_data = pd.DataFrame({
        'latitude':[coordinates[0]],
        'longitude':[coordinates[1]],
        'floor_area_sqm':[input_floor_area_sqm],
        'floor':[input_floor],
        'remaining_lease_years':[input_remaining_lease_years]
    })

    # submit form button
    st.write('Load inputs to machine learning model to prepare for a prediction:')
    submit = st.form_submit_button(label='Load')

# load model
model = pickle.load(open('xgb_baseline.pkl', 'rb'))

# describe predict button
st.write('Take a guess at the price before running the model :)')

# add predict button
if st.button('Predict'):
    # predict input_data using model
    prediction = model.predict(input_data)[0]
    # format prediction with thousands separator and round to two decimal places
    prediction = '{:,}'.format(round(prediction))
    # print prediction
    st.write(f'The predicted resale price of a flat at postal code {input_postal_code} with a floor area of {input_floor_area_sqm} square meters is ${prediction}!')