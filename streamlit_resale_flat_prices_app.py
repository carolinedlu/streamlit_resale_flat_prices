
# conda deactivate

# conda env remove -n streamlit_resale_flat_prices

# conda create -n streamlit_resale_flat_prices python=3.8.5

# conda activate streamlit_resale_flat_prices

# conda install streamlit

# cd C:\Users\Russ\Anaconda3\Russ Projects\Resale Flat Prices\streamlit_resale_flat_prices
# python streamlit_resale_flat_prices_app.py


# imports
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# title of app
st.title('Resale Flat Prices')

# read data
data = pd.read_csv('flat_prices_full.csv')




# display data
st.write(data)

# set plot and figure size
fig, ax = plt.subplots(figsize=(15,10))
# plot ax
ax = sns.boxplot(x='resale_price', y='flat_type', data=data, order=sorted(list(data['flat_type'].unique())))
# set title
ax.set_title('Boxplot of Flat Resale Price by Town')
# show plot
st.pyplot(fig)

# set plot and figure size
fig, ax = plt.subplots(figsize=(15,10))
# plot boxplot
sns.boxplot(x='resale_price', y='town', data=data, order=sorted(list(data['town'].unique())))
# set title
ax.set_title('Boxplot of Flat Resale Price by Town')
# show ploy
st.pyplot(fig)


print('### File ran properly ###')