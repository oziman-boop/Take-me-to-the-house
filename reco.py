import streamlit as st
import pandas as pd
import altair as alt
import folium
import plotly.express as px
from streamlit_folium import st_folium

# Ana veri setini oku
df = pd.read_csv("hemnet_last3.csv")
df['build_year'] = pd.to_datetime(df['build_year']).dt.year

# Popülasyon veri setini oku
population_df = pd.read_csv("sweden_population.csv")
population_df['county'].unique()
# Filtreleme işlevi

#education veri setini oku
education_df = pd.read_csv("sweden_education.csv")

def filter_properties(df, price_min, price_max, property_type, living_area_min, living_area_max, balcony, city, rooms):
    filtered_df = df[
        (df['price'] >= price_min) &
        (df['price'] <= price_max) &
        (df['living_area'] >= living_area_min) &
        (df['living_area'] <= living_area_max) &
        (df['rooms'] >= rooms)
    ]

    if property_type:
        filtered_df = filtered_df[filtered_df['property_type'].str.contains(property_type, case=False)]

    if balcony == 'YES':
        filtered_df = filtered_df[filtered_df['balcony'] == 'Yes']
    elif balcony == 'NO':
        filtered_df = filtered_df[filtered_df['balcony'] != 'Yes']

    if city:
        filtered_df = filtered_df[filtered_df['county'] == city]

    return filtered_df

# Streamlit uygulaması
st.title("HOUSE RECOMMENDATION SYSTEMS")

st.sidebar.header("Filterings")
price_min = st.sidebar.number_input("Minimum fiyat", min_value=0.0, value=0.0, step=10000.0)
price_max = st.sidebar.number_input("Maksimum fiyat", min_value=0.0, value=10000000.0, step=10000.0)

# Mülk tipi için unique değerleri al
property_types = ['Any'] + sorted(df['property_type'].unique())
property_type = st.sidebar.selectbox("Property Type", property_types, index=0)

living_area_min = st.sidebar.number_input("Minimum m2", min_value=0.0, value=0.0, step=1.0)
living_area_max = st.sidebar.number_input("Maximum m2", min_value=0.0, value=1000.0, step=1.0)
balcony = st.sidebar.selectbox("Do you want a balcony?", ["Yes", "No", "It doesn't matter"], index=2)
city = st.sidebar.selectbox("City", ['Any'] + sorted(population_df['county'].unique()))
rooms = st.sidebar.number_input("Minimum Room Number", min_value=0.0, value=1.0, step=1.0)

property_type = None if property_type == "Any" else property_type
balcony = None if balcony == "It doesn't matter" else balcony
city = None if city == "Any" else city

filtered_df = filter_properties(df, price_min, price_max, property_type, living_area_min, living_area_max, balcony, city, rooms)

# İlk 10 evi seç
top_10_df = filtered_df.head(10)

st.header("Recommended Houses")
st.write(top_10_df)

# Fiyat Dağılım Grafiği
st.header("Price Distribution Chart")
price_chart = alt.Chart(top_10_df).mark_bar(strokeWidth=10).encode(
    x='price:Q',
    y='count()'
).properties(
    width=600,
    height=400
)
st.altair_chart(price_chart)

# Harita ve Heatmap ekleme
st.header("Houses on the map and Price Heatmap")

col1, col2 = st.columns(2)

with col1:
    map_data = top_10_df[['latitude', 'longitude', 'area']]
    m = folium.Map(location=[map_data['latitude'].mean(), map_data['longitude'].mean()], zoom_start=10)
    for idx, row in map_data.iterrows():
        folium.Marker([row['latitude'], row['longitude']], popup=row['area']).add_to(m)
    st_folium(m)

with col2:
    heatmap_fig = px.density_mapbox(
        filtered_df, lat='latitude', lon='longitude', z='price', radius=10,
        center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
        zoom=10, mapbox_style="stamen-terrain", title='Price Heatmap'
    )
    st.plotly_chart(heatmap_fig)


# Seçilen şehir için popülasyon grafiği
if city:
    city_population = population_df[population_df['county'].str.contains(city, case=False)]

    if not city_population.empty:
        color_sequence = ['#a3c6c4', '#d9e2e9', '#006400', '#a6d9f7']

        fig_city = px.bar(
            city_population,
            x='age_range',
            y='population',
            color='gender',
            color_discrete_sequence=color_sequence,
            title=f'Population distribution of {city} <br> by age groups',
            labels={'age_range': 'age_range', 'population': 'population'}
        )

        fig_city.update_layout(
            width=1200,
            height=720,
            barmode='group'
        )

# İsveç'in genel popülasyon grafiği
color_sequence = ['#a3c6c4', '#d9e2e9', '#006400', '#a6d9f7']

fig = px.bar(
    population_df,
    x='age_range',
    y='population',
    color='gender',
    color_discrete_sequence=color_sequence,
    title='Population distribution of Sweden <br> by age groups',
    labels={'age_range': 'age_range', 'population': 'population'}
)

fig.update_layout(
    width=1200,
    height=720,
    barmode='group'
)

# Grafikleri yan yana gösterme
col1, col2 = st.columns(2)

with col1:
    if city and not city_population.empty:
        st.plotly_chart(fig_city)
    else:
        st.write("Seçilen şehir için popülasyon verisi mevcut değil.")

with col2:
    st.plotly_chart(fig)

# İsveç'in genel popülasyon grafiği
total_population_df = population_df.groupby('county')['population'].sum().reset_index()

st.header("General Population Distribution of Sweden")
fig_sweden = px.bar(
    total_population_df,
    x='county',
    y='population',
    title='Total population distribution of Sweden <br> by cities',
    labels={'county': 'City', 'population': 'Total Population'}
)

fig_sweden.update_layout(
    width=1200,
    height=720
)

st.plotly_chart(fig_sweden)



#education grafik çizme

#education grafik çizme

st.header("Education Rate by Age Group")

if city:
    df_filtered = education_df[education_df['county'] == city]
    fig_edu = px.bar(
        df_filtered,
        x='age_range',
        y='population',
        title=f'Education rate of {city} by age groups',
        color_discrete_sequence=px.colors.sequential.Plasma_r  # Renk burada değiştirilir
    )
    st.plotly_chart(fig_edu)



# CSV İndirme
csv = top_10_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download recommended homes as CSV",
    data=csv,
    file_name='filtered_properties.csv',
    mime='text/csv',
)
