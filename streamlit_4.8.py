import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from folium import plugins
import geopandas as gpd
from folium.plugins import MarkerCluster

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
import folium
from folium.plugins import HeatMap, MousePosition
import random
from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, \
    mean_squared_log_error, make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
from streamlit_folium import st_folium
import base64
from streamlit_folium import folium_static
import plotly.express as px
import altair as alt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 290)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css('hemnet.css')
pd.set_option('display.max_columns', 500)

@st.cache_data
def load_data():
    df_ = pd.read_csv('hemnet_last.csv')
    df_['sold_date'] = pd.to_datetime(df_['sold_date'], format='%Y-%m-%d')
    df_['build_year'] = pd.to_datetime(df_['build_year'], format='%Y-%m-%d')
    return df_.copy()

df = load_data()


df_coordinates_county = df.groupby('county')[['latitude', 'longitude']].mean().reset_index()
def create_county_markers(lats, lons, counties, location=[62.0, 15.0], zoom_start=4, width=800, height=600):
    """
    Create a map with circle markers for given counties and their coordinates.

    Parameters:
    - lats (list or pd.Series): List or Series of latitudes.
    - lons (list or pd.Series): List or Series of longitudes.
    - counties (list or pd.Series): List or Series of county names.
    - location (list of float): Initial map center coordinates [latitude, longitude]. Default is [62.0, 15.0].
    - zoom_start (int): Initial map zoom level. Default is 4.

    Returns:
    - sweden_map (folium.Map): Folium map object with the county markers.
    """

    if not isinstance(lats, (list, pd.Series)):
        raise ValueError("Lats must be a list or a pandas Series")
    if not isinstance(lons, (list, pd.Series)):
        raise ValueError("Lons must be a list or a pandas Series")
    if not isinstance(counties, (list, pd.Series)):
        raise ValueError("Counties must be a list or a pandas Series")
    if len(lats) != len(lons) or len(lats) != len(counties):
        raise ValueError("Lats, lons, and counties must have the same length")

    if isinstance(lats, pd.Series):
        lats = lats.tolist()
    if isinstance(lons, pd.Series):
        lons = lons.tolist()
    if isinstance(counties, pd.Series):
        counties = counties.tolist()

    sweden_map = folium.Map(location=location, zoom_start=zoom_start, width=width, height=height)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue',
              'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

    for county, lat, lon in zip(counties, lats, lons):
        color = random.choice(colors)
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            popup=f"County: {county},  Lat: {lat}, Lon: {lon}",
            color=color,
            fill=True,
            fillColor=color
        ).add_to(sweden_map)
    return sweden_map

def create_house_count_heatmap(dataframe, selected_county):
    """
    Create an interactive heatmap based on the number of houses using given coordinates grouped by county.
    Includes a mouse position display showing coordinates, markers for the top 3 most populated counties,
    and legends for both the markers and the heatmap colors.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame containing 'county', 'latitude', 'longitude', and 'price' columns.

    Returns:
    - folium.Map: Folium map object with the heatmap, mouse position display, markers, and legends.
    """

    if not all(col in dataframe.columns for col in ['county', 'latitude', 'longitude']):
        raise ValueError("DataFrame must contain 'county', 'latitude', 'longitude' columns")
    df_filtered = dataframe[dataframe['county'] == selected_county]
    mean_lat = df_filtered['latitude'].mean()
    mean_lon = df_filtered['longitude'].mean()
    folium_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)
    marker_cluster = MarkerCluster().add_to(folium_map)
    # Count the number of houses in each coordinate
    heat_data = df_filtered.groupby(['latitude', 'longitude']).size().reset_index(name='count').values.tolist()
    HeatMap(heat_data, radius=15).add_to(folium_map)

    formatter = "function(num) {return L.Util.formatNum(num, 5) + ' º ';};"
    mouse_position = MousePosition(
        position='topright',
        separator=' | ',
        empty_string='NaN',
        lng_first=True,
        num_digits=20,
        prefix="Coordinates:",
        lat_formatter=formatter,
        lng_formatter=formatter
    )
    folium_map.add_child(mouse_position)

    top_3_counties = dataframe['county'].value_counts().nlargest(3).reset_index()
    top_3_counties.columns = ['county', 'count']

    colors = ['red', 'darkred', 'orange']
    for idx, row in top_3_counties.iterrows():
        county_data = dataframe[dataframe['county'] == row['county']].iloc[0]
        folium.Marker(
            location=[county_data['latitude'], county_data['longitude']],
            popup=f"County: {row['county']}<br>Number of Houses: {row['count']}",
            icon=folium.Icon(color=colors[idx], icon='star', prefix='fa')
        ).add_to(folium_map)

    legend_html = '''
     <div style="
     position: fixed;
     bottom: 50px; left: 50px; width: 220px;
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; padding: 10px;
     ">
     <strong>Top 3 Most Populated Counties</strong><br>
     <i class="fa fa-star" style="color: red;"></i> 1st Most Populated<br>
     <i class="fa fa-star" style="color: darkred;"></i> 2nd Most Populated<br>
     <i class="fa fa-star" style="color: orange;"></i> 3rd Most Populated<br>
     <br>
     <strong>House Count Heatmap</strong><br>
     <i style="background: rgba(255, 0, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> High<br>
     <i style="background: rgba(255, 165, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Medium<br>
     <i style="background: rgba(0, 255, 0, 0.6); width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Low<br>
     </div>
     '''
    folium_map.get_root().html.add_child(folium.Element(legend_html))

    return folium_map

st.sidebar.title('🏂 US Population Dashboard')

county_list = df['county'].unique()

selected_county = st.sidebar.selectbox('Please select a county', county_list)

operation = ['House Count Heatmap', 'House Price Heatmap', 'Population Heatmap']
st.sidebar.selectbox('Select operation', operation)

if operation == 'House Count Heatmap':
    heatmap = create_house_count_heatmap(df, selected_county)
    folium_static(heatmap)











