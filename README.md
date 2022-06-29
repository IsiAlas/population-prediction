# population-prediction

**Objective of the Project**

- Predicting the population per country from 2021 - 2050, using historical data from 1950 - 2020 from the [United Nations](https://population.un.org/wpp/Download/Standard/CSV/).
- Sharing the outcomes through interactive dashboards using [Flourish](https://flourish.studio/) as the visualisation tool.

**Results**

Using the file predict.py, the result is a csv file with the population prediction from 2021 - 2050 for 223 countries and locations.

- Additionally, we have created some [interactive visualizations](https://www.notion.so/Population-Prediction-Interactive-Visualisationsfc394bbd153a49e198773528b8f75093) with the prediction results, along wiht the historical data.
- You can find our results in forecast_per_country_2021-2050.csv

**Dataset description**

Total population by sex, annually from 1950 to 2100.

- LocID (numeric): numeric code for the location; for countries and areas, it follows the ISO 3166-1 numeric standard
- Location (string): name of the region, subregion, country or area
- VarID (numeric): numeric code for the variant
- Variant (string): projection variant name (Medium is the most used)
- Time (string): label identifying the single year (e.g. 1950) or the period of the data (e.g. 1950-1955)
- MidPeriod (numeric): numeric value identifying the mid period of the data, with the decimal representing the month (e.g. 1950.5 for July 1950)
- PopMale: Total male population (thousands)
- PopFemale: Total female population (thousands)
- PopTotal: Total population, both sexes (thousands)
- PopDensity: Population per square kilometre (thousands)

**NOTE:** Data from 1950 - 2022 is historical data. 

**Dataset Preprocessing**

- Filtered the dataset to only have data from 1950 - 2020 (historical data) with the Medium Variant.
- More details on the preprocessing in the notebook timeseries_forecasts.ipynb

**Prediction Process**

For making the predictions we used Time Series Modelling, specifically the Auto Regressive Integrated Moving Average (ARIMA) method. 

We used the `auto_arima` [function](http://alkaline-ml.com/pmdarima/about.html#about) from the `pmdarima` Python library, that works similarly to a grid search where it selects the optimal parameters that minimise the **Akaike Information Criterion** (AIC), which is a performance metric.

The parameters optimised are:

- Differentiating level - ”d”
- Partial Autocorrelation lags statistically significant - “p”
- Autocorrelation lags statistically significant - “q”

