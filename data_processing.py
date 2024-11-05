import numpy as np
import pandas as pd
import os

# data processing to extract price information from csv files and create features

shorter_period = 5
medium_period = 30
longer_period = 90
start_date = '2005-01-01'
end_date = '2021-12-31'

stocks_array = []
first = True
for file in os.listdir('csv_data'):
    loop_df = pd.read_csv(f'csv_data/{file}', usecols=['Date', 'Adjusted Close'], parse_dates=['Date'], dayfirst=True).set_index('Date')
    is_null = loop_df.isnull().values.any()
    repeat_prices = loop_df['Adjusted Close'].value_counts().max()
    start_year = int(list(loop_df.index)[0].year)
    end_year = int(list(loop_df.index)[-1].year)
    if is_null or start_year > 2005 or end_year < 2022 or repeat_prices > 50:
        continue
    if first:
        loop_df_dated = loop_df.loc[start_date:end_date]
        dates = loop_df_dated.index
        stocks_array.append(np.array(loop_df_dated['Adjusted Close']))
        first = False
        continue
    exists = dates.isin(loop_df.index)
    if exists.all() and not is_null:
        loop_df_dated = loop_df.loc[dates]
        stocks_array.append(np.array(loop_df_dated['Adjusted Close']))

dates = pd.to_datetime(dates).strftime('%d/%m/%y')

stocks_array = np.array(stocks_array)

return_array = ( stocks_array[:, 1:] - stocks_array[:, :-1] ) / stocks_array[:, :-1]

average_return = np.mean(return_array, axis=0)

short_correlations = []
medium_correlations = []
long_correlations = []

short_volatilities = []
medium_volatilities = []
long_volatilities = []

short_returns = []
medium_returns = []
long_returns = []

for day in range(longer_period,len(average_return)+1):

    stocks_last_5_days_returns = return_array[:, day-shorter_period:day]
    market_last_5_days_returns = average_return[day-shorter_period:day]

    short_returns.append((stocks_array[:, day] - stocks_array[:, day-shorter_period]) / stocks_array[:, day-shorter_period])
    short_correlation = np.array([np.corrcoef(row, market_last_5_days_returns)[0, 1] for row in stocks_last_5_days_returns])
    short_correlations.append(short_correlation)
    short_volatilities.append(np.std(stocks_last_5_days_returns, axis=1))

    stocks_last_30_days_returns = return_array[:, day-medium_period:day]
    market_last_30_days_returns = average_return[day-medium_period:day]

    medium_returns.append((stocks_array[:, day] - stocks_array[:, day-medium_period]) / stocks_array[:, day-medium_period])
    medium_correlation = np.array([np.corrcoef(row, market_last_30_days_returns)[0, 1] for row in stocks_last_30_days_returns])
    medium_correlations.append(medium_correlation)
    medium_volatilities.append(np.std(stocks_last_30_days_returns, axis=1))

    stocks_last_90_days_returns = return_array[:, day-longer_period:day]
    market_last_90_days_returns = average_return[day-longer_period:day]

    long_returns.append((stocks_array[:, day] - stocks_array[:, day-longer_period]) / stocks_array[:, day-longer_period])
    long_correlation = np.array([np.corrcoef(row, market_last_90_days_returns)[0, 1] for row in stocks_last_90_days_returns])
    long_correlations.append(long_correlation)
    long_volatilities.append(np.std(stocks_last_90_days_returns, axis=1))

split = int(4280 * 0.6)

train_stocks_array = stocks_array[:, :split]
train_dates = dates[:split]
train_short_correlations = short_correlations[:split]
train_medium_correlations = medium_correlations[:split]
train_long_correlations = long_correlations[:split]
train_short_volatilities = short_volatilities[:split]
train_medium_volatilities = medium_volatilities[:split]
train_long_volatilities = long_volatilities[:split]
train_short_returns = short_returns[:split]
train_medium_returns = medium_returns[:split]
train_long_returns = long_returns[:split]

test_stocks_array = stocks_array[:, split:]
test_dates = dates[split:]
test_short_correlations = short_correlations[split:]
test_medium_correlations = medium_correlations[split:]
test_long_correlations = long_correlations[split:]
test_short_volatilities = short_volatilities[split:]
test_medium_volatilities = medium_volatilities[split:]
test_long_volatilities = long_volatilities[split:]
test_short_returns = short_returns[split:]
test_medium_returns = medium_returns[split:]
test_long_returns = long_returns[split:]

np.save('train_arrays/train_stocks_array.npy', train_stocks_array)
np.save('train_arrays/train_dates.npy', np.array(train_dates))
np.save('train_arrays/train_short_correlations.npy', np.array(train_short_correlations))
np.save('train_arrays/train_medium_correlations.npy', np.array(train_medium_correlations))
np.save('train_arrays/train_long_correlations.npy', np.array(train_long_correlations))
np.save('train_arrays/train_short_volatilities.npy', np.array(train_short_volatilities))
np.save('train_arrays/train_medium_volatilities.npy', np.array(train_medium_volatilities))
np.save('train_arrays/train_long_volatilities.npy', np.array(long_correlations))
np.save('train_arrays/train_short_returns.npy', np.array(train_short_returns))
np.save('train_arrays/train_medium_returns.npy', np.array(train_short_returns))
np.save('train_arrays/train_long_returns.npy', np.array(train_long_returns))

np.save('test_arrays/test_stocks_array.npy', test_stocks_array)
np.save('test_arrays/test_dates.npy', np.array(test_dates))
np.save('test_arrays/test_short_correlations.npy', np.array(test_short_correlations))
np.save('test_arrays/test_medium_correlations.npy', np.array(test_medium_correlations))
np.save('test_arrays/test_long_correlations.npy', np.array(test_long_correlations))
np.save('test_arrays/test_short_volatilities.npy', np.array(test_short_volatilities))
np.save('test_arrays/test_medium_volatilities.npy', np.array(test_medium_volatilities))
np.save('test_arrays/test_long_volatilities.npy', np.array(long_correlations))
np.save('test_arrays/test_short_returns.npy', np.array(test_short_returns))
np.save('test_arrays/test_medium_returns.npy', np.array(test_short_returns))
np.save('test_arrays/test_long_returns.npy', np.array(test_long_returns))
