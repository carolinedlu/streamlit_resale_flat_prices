'''
Overview

'''

# imports
import pandas as pd
import numpy as np
import datetime as dt
import requests
import json
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_validate


# combine, clean, and transform resale flat prices data
def combine_clean_transform(list_of_flat_prices_df, coordinates_map, list_of_columns_to_use, start_date='1990-01-01', end_date='2100-01-01'):
    '''
    combine, clean, and transform resale flat prices data
    combine multiple dataframes together
    filter date range of data to be used
    clean 
        flat_model
        flat_type
        street_name
    create 
        floor from storey_range,
        remaining_lease_years from lease_commence_date
        coordinates
    select columns to be used for analysis

    arguments:
    list_of_flat_prices_df (list): list of dataframes of resale flat prices
    coordinates_map (df): dataframe of address and coordinates to map coordinates to address
    list_of_columns_to_use (list): list of columns to used for analysis
    start_date (str): default to '1990-01-01', start date of date range to be used for analysis in yyyy-mm-dd format, e.g. '1990-01-01' 
    end_date (str): default to '2100-01-01', end date of date range to be used for analysis in yyyy-mm-dd format, e.g. '2021-06-01'

    returns:
    a dataframe with columns from list_of_columns_to_use
    '''
    # merge all flat prices together
    flat_prices_clean = pd.concat(list_of_flat_prices_df)

    # convert month to datetime format
    flat_prices_clean['year_month'] = pd.to_datetime(flat_prices_clean['month'])
    # filter max df date for analysis
    flat_prices_clean = flat_prices_clean.loc[(flat_prices_clean['year_month'] >= start_date) & (flat_prices_clean['year_month'] <= end_date)]

    # clean and engineer flat_prices_clean data
    # convert flat_model to upper for consistency
    flat_prices_clean['flat_model'] = flat_prices_clean['flat_model'].str.upper()
    # convert 'MULTI-GENERATION' and 'MULTI GENERATION' to 'MULTI GENERATION' for consistency
    flat_prices_clean['flat_type'] = flat_prices_clean['flat_type'].str.replace('-', ' ')
    # replace 'C'WEALTH' with 'COMMONWEALTH' in street_name for better results when fetching coordinates
    flat_prices_clean['street_name'] = flat_prices_clean['street_name'].str.replace('C\'WEALTH', 'COMMONWEALTH')

    # get max floor of storey_range as new 'floor' column and covert to int64
    # 'floor' will become a numerical data, a higher floor is typically considered to be better.
    flat_prices_clean['floor'] = flat_prices_clean['storey_range'].str[-2:].astype('int64')
    # create new column remaining_lease_years to show as number of years only
    # calculated as current year minus lease_commence_date
    flat_prices_clean['remaining_lease_years'] = dt.datetime.now().year - flat_prices_clean['lease_commence_date']
    # merge coordinates on address
    # create full address from block and street_name
    flat_prices_clean['full_address'] = flat_prices_clean['block'] + ' ' + flat_prices_clean['street_name'] + ' SINGAPORE'

    # merge latitude and longitude
    flat_prices_clean = pd.merge(flat_prices_clean, coordinates_map[['full_address', 'latitude', 'longitude']], how='left', on='full_address')

    # select columns to use
    flat_prices_clean = flat_prices_clean[list_of_columns_to_use]

    # return
    return flat_prices_clean


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


# clean coordinates
def clean_coordinates(coordinates_df, coordinates_boundary):
    '''
    clean coordinates by removing results from geocode api with errors
    remove coordinates that do not fall within a defined boundary, e.g. coordinates should not be outside of singapore

    arguments:
    coordinates_df (df): dataframe of address and coordinates after applying get_coordinates_from_address()
    coordinates_boundary (dict): nested dictionary of boundary of selected country with minimum and maximum latitude and longitude e.g. coordinates_boundary['SG']

    returns:
    a dataframe
    '''
    # copy data
    coordinates_clean = coordinates_df.copy()

    # split coodinates to latitude and longitude
    coordinates_clean['latitude'], coordinates_clean['longitude'] = zip(*coordinates_clean['coordinates'])
    # remove records where there are errors in coordinates
    coordinates_clean = coordinates_clean.loc[~(coordinates_clean['latitude'].astype(str).str.contains('error'))]
    # convert latitude and longitude to numeric
    coordinates_clean[['latitude', 'longitude']] = coordinates_clean[['latitude', 'longitude']].apply(pd.to_numeric)
    # filter coordinates
    coordinates_clean = coordinates_clean.loc[
        (coordinates_clean['latitude'] >= coordinates_boundary['min_lat']) &
        (coordinates_clean['latitude'] <= coordinates_boundary['max_lat']) &
        (coordinates_clean['longitude'] >= coordinates_boundary['min_lon']) &
        (coordinates_clean['longitude'] <= coordinates_boundary['max_lon'])
        ]

    # return
    return coordinates_clean

# validate model using kfolds cross validation
def cv_results(model, X, y, num_folds, kfold_random_state):
    '''
    validate model using kfolds cross validation

    arguments:
    model (model): model to be used for prediction
    X (df): dataframe of independent variables
    y (df): dataframe of dependent variables
    num_folds (int): number of folds to use

    returns:
    printed statement
    '''
    # define number of folds and shuffle data
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=kfold_random_state)
    # apply cross validation to return train/test score of mae and rmse
    cvresults = cross_validate(model, X, y, cv=kf, return_train_score=True, scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error'))
    # print model hyperparameters
    print(f'model name: {model}')
    # print mean train mae
    print('Train MAE: {}'.format(abs(cvresults['train_neg_mean_absolute_error'].mean())))
    # print mean validation mae
    print('Validation MAE: {}'.format(abs(cvresults['test_neg_mean_absolute_error'].mean())))
    # print mean train rmse
    print('Train RMSE: {}'.format(abs(cvresults['train_neg_root_mean_squared_error'].mean())))
    # print mean validation rsme
    print('Validation RMSE: {}'.format(abs(cvresults['test_neg_root_mean_squared_error'].mean())))

    # return
    return cvresults



# get feature importance of xgb model
def get_feature_importance_from_xgb(xgb_model, X):
    '''
    get feature importance of weight and gain of features used in xgb model as one dataframe

    arguments:
    xgb_model (model): fitted xgb model
    X (df): dataframe of dependent variables to extract features

    returns:
    a dataframe with each feature's weight and gain
    '''
    # create dataframe with features as column
    xgb_feature_importance = pd.DataFrame({'features':list(X)})

    # get feature importances
    xgb_feature_importance_weights = xgb_model.get_booster().get_score(importance_type='weight')
    xgb_baseline_importance_gain = xgb_model.get_booster().get_score(importance_type='gain')

    # map feature importances to dataframe
    xgb_feature_importance['weight'] = xgb_feature_importance['features'].map(xgb_feature_importance_weights)
    xgb_feature_importance['gain'] = xgb_feature_importance['features'].map(xgb_baseline_importance_gain)

    # return 
    return xgb_feature_importance



# plot feature importance from xgb model
def plot_feature_importance_from_xgb(xgb_feature_importance, figsize, file_name='False'):
    '''
    plot feature importance of weight and gain from xgb model

    arguments:
    xgb_feature_importance (df): a dataframe of features and their weight and gain from get_feature_importance_from_xgb()
    figsize (tuple): a tuple of figsize to adjust, e.g. (20,5)
    file_name (str): default as 'False', if file_name entered then save file as file_name

    returns:
    show plot inline with optional saved file
    '''
    # set subplot sizes
    plt.subplots(figsize=(figsize))

    # weight of features
    plt.subplot(1, 2, 1)
    # plot barplot
    sns.barplot(data=xgb_feature_importance, x='weight', y='features')
    # set title
    plt.title('XGB Feature Importance: Weight')

    # gain of features
    plt.subplot(1, 2, 2)
    # plot barplot
    sns.barplot(data=xgb_feature_importance, x='gain', y='features')
    # set title
    plt.title('XGB Feature Importance: Gain')

    # save file if file_name has been entered
    if file_name != 'False':
        plt.savefig(file_name + '.png')

    # show plot
    plt.show()



# predict using model and print mean absolute error and root mean squared error
def pred_model(model, X_test, y_test):
    '''
    predict using model and print metrics

    arguments:
    model (model): fitted model to be used for prediction
    X_test (df): dataframe of independent variables from test data
    y_test (series): series of the dependent variable from test data

    returns:
    printed statement and predictions
    '''
    # predict using trained model
    y_pred = model.predict(X_test)
    # calculate and print mae
    print ('Test MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
    # caclulate and print rmse
    print ('Test RMSE: {}'.format(mean_squared_error(y_test, y_pred, squared=False)))

    # return
    return y_pred