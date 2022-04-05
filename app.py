import streamlit as st
from gsheetsdb import connect
import pandas as pd
import datetime
import numpy as np


conn = connect()

sheet_url = 'https://docs.google.com/spreadsheets/d/1JDgPZDWubSUz7qHjYjfT0G0JwP9DXjVzXYV5Y_8xckw/edit?usp=sharing'

# sheet_url = st.secrets["public_gsheets_url"] this was used if you want to hide db info

# @st.cache()
# def dataframe10():
#     return pd.read_sql(f'SELECT * FROM "{sheet_url}" LIMIT 10', conn)

st.write("This is not finished project yet. It has some bugs (when filter value is not in the table, for example).\n"
         " In the future it is thought to add machine learning prediction models (models are almost ready) instead of using simple averages according to filters."
         " Although some already can find suitable homes with low price (even for reselling)"
         )

@st.cache(ttl=60*60*24)
def dataframe():
    data = pd.read_sql(f'''SELECT ad_number,prices,categories,floors,areas_m2,rooms,credits,district,
      ownerships,watches,ad_refreshed_date,date_of_parsing,agency_titles FROM "{sheet_url}" ''', conn)

    data['prices'] = pd.to_numeric(
        data['prices'].apply(lambda x: str(x).replace(" ", ""))
    )

    data['watches'] = data['watches'].apply(lambda x: int(x))

    apartment_floor = data['floors'].apply(lambda x: str(x).split(" / ")[0])
    buildings_floor = []
    for i in data['floors']:
        try:
            buildings_floor.append(str(i).split(" / ")[1])
        except:
            buildings_floor.append(np.nan)
    data['apartment_floor'] = apartment_floor
    data['buildings_floor'] = buildings_floor

    data['areas_m2'] = data['areas_m2'].apply(lambda x: x.rstrip(" m²"))
    data['areas_m2'] = data['areas_m2'].apply(lambda x: x.rstrip(" sot"))
    data['areas_m2'] = data['areas_m2'].apply(lambda x: round(float(x)))     
    data['rooms'] = pd.to_numeric(data['rooms'])
    data['watches'] = pd.to_numeric(data['watches'])

    data['ad_refreshed_date_eng'] = data['ad_refreshed_date'].apply(
        lambda x: x.replace("Yanvar", "-01-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace("Fevral", "-02-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace("Mart", "-03-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace("Aprel", "-04-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace("May", "-05-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace("İyun", "-06-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace("İyul", "-07-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace("Avqust", "-08-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace("Sentyabr", "-09-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace("Oktyabr", "-10-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace("Noyabr", "-11-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace("Dekabr", "-12-")
    )
    data['ad_refreshed_date_eng'] = data['ad_refreshed_date_eng'].apply(
        lambda x: x.replace(" ", "")
    )

    ad_refreshed_date_eng_final = []

    for index, value in enumerate(data['ad_refreshed_date_eng']):
        if "Bugün" in value:
            ad_refreshed_date_eng_final.append(data.iloc[index]['date_of_parsing'].date())
        elif "Dünən" in value:
            ad_refreshed_date_eng_final.append(data.iloc[index]['date_of_parsing'].date() - datetime.timedelta(days=1))
        else:
            ad_refreshed_date_eng_final.append(value)

    data['ad_refreshed_date_eng_final'] = pd.Series(ad_refreshed_date_eng_final).apply(lambda x: str(x))
    data['ad_refreshed_date_eng_final'] = pd.to_datetime(data['ad_refreshed_date_eng_final'])

    # df['date_ad_refreshed'] = pd.Series(ad_refreshed_date_eng_final).apply(
    #     lambda x: x if type(x) is datetime.date else datetime.datetime.strptime(x, "%d %B %Y ").date()
    # )
    data['price_per_meter2'] = round(data['prices'] / data['areas_m2'])
    data = data.dropna()
    data['ad_number'] = data['ad_number'].apply(lambda x: str(x))
    data['date_of_parsing'] = data['date_of_parsing'].apply(lambda x: x.date())

    return data


preprocessed_data = dataframe()

@st.cache
def district(preprocessed_data = preprocessed_data):
    districts = []
    districts_in_lists = list(preprocessed_data['district'].apply(lambda x: x.split("; ")))
    for i in districts_in_lists:
        for x in i:
            districts.append(x)

    return sorted(list(set(districts)))

districts = district()

district = st.sidebar.selectbox("Rayonlar", districts)
watches = st.sidebar.slider("Baxışların sayı",value = [100, max(preprocessed_data['watches'])],
                            step=1, min_value=1, max_value=int(preprocessed_data['watches'].max()))
rooms = st.sidebar.slider("Otaq sayı", step=1,
                          min_value=round(preprocessed_data['rooms'].quantile(0.01)),
                          max_value=round(preprocessed_data['rooms'].quantile(0.99)),
                          value=[round(preprocessed_data['rooms'].quantile(0.01)),
                                 round(preprocessed_data['rooms'].quantile(0.99))]
                          )
prices = st.sidebar.slider("Qiymət aralığı", step=1,
                           min_value=round(preprocessed_data['prices'].quantile(0.01)),
                           max_value=round(preprocessed_data['prices'].quantile(0.99)),
                           value=[round(preprocessed_data['prices'].quantile(0.01)),
                                  round(preprocessed_data['prices'].quantile(0.99))])
price_per_meter2 = st.sidebar.slider("Mənzilin kvadrat metrin qiyməti", step=1,
                          min_value=round(preprocessed_data['price_per_meter2'].quantile(0.01)),
                          max_value=round(preprocessed_data['price_per_meter2'].quantile(0.99)),
                          value=[round(preprocessed_data['price_per_meter2'].quantile(0.01)),
                                 round(preprocessed_data['price_per_meter2'].quantile(0.99))])
areas = st.sidebar.slider("Mənzilin ümumi sahəsi", step=1,
                          min_value=round(preprocessed_data['areas_m2'].quantile(0.01)),
                          max_value=round(preprocessed_data['areas_m2'].quantile(0.99)),
                          value=[round(preprocessed_data['areas_m2'].quantile(0.01)),
                                 round(preprocessed_data['areas_m2'].quantile(0.99))])
# st.sidebar.date_input("Elanın tarixi", min_value = min(preprocessed_data['ad_refreshed_date_eng_final']),
#                       max_value = max(preprocessed_data['ad_refreshed_date_eng_final']))



filtered_data = preprocessed_data[
             (preprocessed_data['watches']   >=watches[0]) &
             (preprocessed_data['watches']   <=watches[1]) &
             (preprocessed_data['rooms']     >=rooms[0]) &
             (preprocessed_data['rooms']     <=rooms[1]) &
             (preprocessed_data['prices']    >=prices[0]) &
             (preprocessed_data['prices']    <=prices[1]) &
             (preprocessed_data['areas_m2']  >=areas[0]) &
             (preprocessed_data['price_per_meter2']  <price_per_meter2[1]) &
             (preprocessed_data['price_per_meter2']    >=price_per_meter2[0]) &
             (preprocessed_data['prices']    <=prices[1]) &
             (preprocessed_data['district'].str.contains(district))
         ]

mean_price_per_m2 = round(filtered_data['price_per_meter2'].mean())

st.title(f"Seçilən paramerlər üzrə ortalama m² qiymət {mean_price_per_m2} AZN")

filtered_data['profit'] = round(
mean_price_per_m2 * filtered_data['areas_m2'] - filtered_data['prices']
)
filtered_data = filtered_data.sort_values(by='profit', ascending=False)
filtered_data['price_per_meter2'] = filtered_data['price_per_meter2'].apply(lambda x: str(x))

data_to_show = filtered_data[['prices', 'price_per_meter2', 'areas_m2', 'categories', 'profit',
                              # 'ad_refreshed_date',
                              # 'date_of_parsing',
                              'ad_number']]
data_to_show['ad_number'] = data_to_show['ad_number'].apply(lambda x: f'https://bina.az/items/{x.rstrip(".0")}')

#st.table(filtered_data)

data_to_show['ad_number'] = data_to_show['ad_number'].apply(lambda x: f'<a target="_blank" href="{x}">Keçid</a>')
data_to_show.columns = ['Mənzilin qiyməti', "m² qiyməti", "Ümumi sahəsi", "Kateqoriya", "Ortalama qazanc", "Keçid"]

data_to_show = data_to_show.to_html(escape=False)
st.write(data_to_show, unsafe_allow_html=True)


