"""
Authors:
- Veronica Herrera G. (VeronicaHG)
- Isidora Anabalon G. (IsiAlas)

Created Date: 29/06/2022
Time Series model to predict a country population.


"""
#Data Manipulation
import numpy as np
import pandas as pd

#List of countries
import pycountry

#Data Visualization
import matplotlib.pyplot as plt
import plotly.express as px

#Time Series
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# List with all the countries

country_list = ['Afghanistan','Albania','Algeria','American Samoa','Andorra','Angola','Antigua and Barbuda','Argentina','Armenia','Aruba','Australia','Austria','Azerbaijan',
 'Bahamas','Bahrain','Bangladesh','Barbados','Belarus','Belgium','Belize','Benin','Bermuda','Bhutan', "Bolivia (Plurinational State of)",'Bosnia and Herzegovina',
 'Botswana','Brazil','Brunei Darussalam','Bulgaria','Burkina Faso','Burundi','Cabo Verde','Cambodia','Cameroon', 'Canada', 'Cayman Islands','Central African Republic','Chad',
 'Chile','China','Colombia','Comoros','Congo','Democratic Republic of the Congo','Cook Islands','Costa Rica','Croatia','Cuba','Curaçao','Cyprus','Czechia',"Côte d'Ivoire",'Denmark',
 'Djibouti','Dominica','Dominican Republic','Ecuador','Egypt','El Salvador','Equatorial Guinea','Eritrea','Estonia','Eswatini','Ethiopia','Falkland Islands (Malvinas)',
 'Faroe Islands','Fiji','Finland','France','French Guiana','French Polynesia','Gabon','Gambia','Georgia','Germany','Ghana','Gibraltar','Greece','Greenland','Grenada',
 'Guadeloupe', 'Guam','Guatemala','Guinea','Guinea-Bissau', 'Guyana','Haiti','Holy See','Honduras','Hungary','Iceland','India',
 'Indonesia','Iran (Islamic Republic of)','Iraq', 'Ireland','Isle of Man','Israel','Italy','Jamaica','Japan','Jordan','Kazakhstan','Kenya','Kiribati',
 "Dem. People's Republic of Korea",'Republic of Korea','Kuwait','Kyrgyzstan',"Lao People's Democratic Republic",'Latvia','Lebanon','Lesotho','Liberia',
 'Libya','Liechtenstein','Lithuania','Luxembourg','Madagascar','Malawi','Malaysia','Maldives','Mali','Malta','Marshall Islands','Martinique','Mauritania','Mauritius','Mayotte',
 'Mexico','Monaco','Mongolia','Montenegro','Montserrat','Morocco','Mozambique','Myanmar', 'Namibia','Nauru','Nepal','Netherlands','New Caledonia','New Zealand',
 'Nicaragua','Niger','Nigeria','Niue','North Macedonia','Northern Mariana Islands','Norway', 'Oman', 'Pakistan','Palau','State of Palestine','Panama','Papua New Guinea',
 'Paraguay','Peru','Philippines','Poland','Portugal', 'Puerto Rico','Qatar','Romania','Russian Federation','Rwanda','Réunion','Saint Barthélemy','Saint Kitts and Nevis',
 'Saint Lucia','Saint Martin (French part)','Saint Pierre and Miquelon','Saint Vincent and the Grenadines','Samoa','San Marino','Sao Tome and Principe','Saudi Arabia','Senegal','Serbia',
 'Seychelles','Sierra Leone','Singapore','Sint Maarten (Dutch part)','Slovakia','Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan', 'Spain',
 'Sri Lanka', 'Sudan','Suriname','Sweden','Switzerland','Syrian Arab Republic','Tajikistan','United Republic of Tanzania','Thailand','Timor-Leste','Togo','Tokelau',
 'Tonga','Trinidad and Tobago', 'Tunisia','Turkey','Turkmenistan','Turks and Caicos Islands','Tuvalu','Uganda','Ukraine','United Arab Emirates','United Kingdom','United States of America',
 'Uruguay','Uzbekistan','Vanuatu','Venezuela (Bolivarian Republic of)','Viet Nam','Western Sahara','Yemen','Zambia','Zimbabwe']

def prediction(df, country_list, start_year, end_year):
    '''Makes forecasts using ARIMA method, returns a dictionary with per country predictions'''

    #Creating empty dictionary where the forecast will be added
    forecasts_dict = {}

    #Looping thought the countries list to apply the model
    for i in country_list:
        df_pred = df[df.Location==i][['Time','PopTotal']]
        df_pred.set_index('Time', inplace= True)
        smodel = pm.auto_arima(df_pred,
                           seasonal=False,
                           start_p=0,
                           max_p=4,
                           max_d=3,
                           start_q=0,
                           max_q=4,
                           trace=True,
                           error_action='ignore')

        #1. Initialize model
        order = smodel.get_params()['order']
        arima = ARIMA(df_pred, order=order)

        #2. Fit the model
        arima = arima.fit()
        forecast = arima.forecast((end_year-start_year), alpha=0.05)

        # 3. Adding forecast per country to the dictionary
        forecasts_dict[i] = forecast

        #4. Transformig the dictionary to a DataFrame
        #df_forecasts=pd.DataFrame.from_dict(forecasts_dict)
        df_forecasts=pd.DataFrame(forecasts_dict)

        #5. Adding the year of the prediction as the index
        df_forecasts.index = range(start_year, end_year, 1)

    return df_forecasts

def to_csv(df, file_name):
    '''Transforms DataFrame to csv file'''

    df.T.to_csv(f'/Users/isi/code/IsiAlas/population-prediction/{file_name}.csv', sep=";", decimal=",")

def main():

    # Load the Dataset
    df_pop = pd.read_csv('/Users/isi/code/IsiAlas/population-prediction/WPP2019_TotalPopulationBySex_Original.csv',
                     sep=";",
                     decimal=",")

    #Filtering the DataFrame to Variant "Medium"
    df_pop_medium = df_pop[df_pop['Variant']=='Medium']

    # Filtering the DataFrame so it only has the historical data, that is 1950 -2020
    df_pop_medium.drop(df_pop_medium.loc[df_pop_medium['Time']>2020].index, inplace=True)

    # Reset index
    df_pop_medium = df_pop_medium.reset_index(drop=True)

    results = prediction(df_pop_medium , country_list, 2021, 2050)

    to_csv(results, f'forecast_per_country_2021-2051')

if __name__ == "__main__":
    main()
