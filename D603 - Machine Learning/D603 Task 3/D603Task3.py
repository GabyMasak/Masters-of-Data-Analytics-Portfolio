import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import shapiro, norm, probplot
import itertools

# Read data from a CSV file
file_path = (r"C:\Users\gabri\OneDrive\Documents\Education\WGU\MSDA\D603 - "
             r"Machine Learning\D603 Task 3\medical_clean.csv")
revData = pd.read_csv(file_path)

# Data Cleaning
# Change the first date to zero
revData.iloc[0,0] = 0

# Change the column headers to 'DATE' and 'REVENUE'
revData.columns = ['DATE', 'REVENUE']

# Convert DATE column to numeric
revData['DATE'] = pd.to_numeric(revData['DATE'])

# Make the `DATE` column the new index
revData.set_index('DATE', inplace=True)

# Check for missing data
print("If there were no missing rows, there would be 731 rows of minute data")
print("The actual length of the DataFrame is:", len(revData))
if len(revData) == 731:
    print("There are no missing values.\n")
else:
    # Everything
    set_everything = set(range(731))

    # The intraday index as a set
    set_revenue = set(revData.index)

    # Calculate the difference
    set_missing = set_everything - set_revenue

    # Print the difference
    print("Missing rows: ", set_missing)

    # Fill in the missing rows
    revData = revData.reindex(range(731), method='ffill')

# Change the index to the days
revData.index = pd.date_range(start='2023-01-01', periods=len(revData), freq='D')

# Plot the original time series
plt.figure(figsize=(12, 4))
plt.plot(revData.index, revData['REVENUE'], label='Original Time Series')
plt.title('Original Time Series')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()

# Track the number of differencing steps
differencing_steps = 0

# Function to check the stationarity of a time series
def check_stationarity(series, significance_level=0.05):
    result = adfuller(series.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    return result[1] <= significance_level

# Apply differencing and recheck for stationarity until the data becomes stationary
if not check_stationarity(revData):
    print("The time series is not stationary. Applying differencing.")
    revData_diff= revData.diff()
    revData_diff.dropna(inplace=True)
    differencing_steps += 1

# Plot the differenced time series
plt.figure(figsize=(12, 4))
plt.plot(revData_diff.index, revData_diff['REVENUE'], label='Differenced Time Series')
plt.title('Differenced Time Series')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()

# Check for normality of the differenced data
stat, p = shapiro(revData_diff['REVENUE'])
print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Differenced data follows a normal distribution (fail to reject H0)')
else:
    print('Differenced data does not follow a normal distribution (reject H0)')

# Plot the histogram and Q-Q plot of the differenced data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(revData_diff['REVENUE'], bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, revData_diff['REVENUE'].mean(), revData_diff['REVENUE'].std())
plt.plot(x, p, 'k', linewidth=2)
plt.title('Histogram of Differenced Data')
plt.xlabel('Revenue')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
probplot(revData_diff['REVENUE'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Differenced Data')

plt.tight_layout()
plt.show()

# Plot the ACF and PACF on the differenced dataset
fig, axes = plt.subplots(2,1)
# Plot the ACF
plot_acf(revData_diff, lags=20, ax=axes[0])
# Plot the PACF
plot_pacf(revData_diff, lags=20, ax=axes[1])
plt.show()

# Plot the spectral density
frequencies, spectrum = periodogram(revData_diff['REVENUE'])
plt.figure(figsize=(12, 4))
plt.semilogy(frequencies, spectrum)
plt.title('Spectral Density')
plt.xlabel('Frequency')
plt.ylabel('Spectrum')
plt.grid(True)
plt.show()

# Apply FFT to the differenced data
fft_result = np.fft.fft(revData_diff['REVENUE'])
freq = np.fft.fftfreq(len(revData_diff['REVENUE']))

# Plot the FFT result
plt.figure(figsize=(12, 4))
plt.plot(freq, np.abs(fft_result))
plt.title('FFT of Differenced Data')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(revData_diff, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(revData_diff, label='Differenced Time Series')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Confirm lack of trends in the residuals
plt.figure(figsize=(12, 4))
plt.plot(residual, label='Residuals')
plt.title('Residuals of Decomposed Series')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()

# Split the original data into training and test sets
train_size = int(len(revData) * 0.8)
train, test = revData.iloc[:train_size], revData.iloc[train_size:]

# Perform grid search to determine the best parameters for ARIMA model on training data
p = range(0, 5)
d = differencing_steps
q = range(0, 5)
pdq = list(itertools.product(p, [d], q))

best_aic_train = np.inf
best_pdq_train = None
best_model_train = None

for param in pdq:
    try:
        model = ARIMA(train, order=param)
        results = model.fit()
        if results.aic < best_aic_train:
            best_aic_train = results.aic
            best_pdq_train = param
            best_model_train = results
    except:
        continue

print(f'Best ARIMA model for training data: ARIMA{best_pdq_train} - AIC:{best_aic_train}')
print(best_model_train.summary())

# Calculate residuals and mean absolute error for training data
residuals_train = best_model_train.resid
mae_train = np.mean(np.abs(residuals_train))
print('Mean absolute error for training data:', mae_train)

# Perform grid search to determine the best parameters for ARIMA model on test data
best_aic_test = np.inf
best_pdq_test = None
best_model_test = None

for param in pdq:
    try:
        model = ARIMA(test, order=param)
        results = model.fit()
        if results.aic < best_aic_test:
            best_aic_test = results.aic
            best_pdq_test = param
            best_model_test = results
    except:
        continue

print(f'Best ARIMA model for test data: ARIMA{best_pdq_test} - AIC:{best_aic_test}')
print(best_model_test.summary())

# Calculate residuals and mean absolute error for test data
residuals_test = best_model_test.resid
mae_test = np.mean(np.abs(residuals_test))
print('Mean absolute error for test data:', mae_test)

# Forecast using the best model for training data
diff_forecast = best_model_train.get_forecast(steps=180)
mean_forecast = diff_forecast.predicted_mean
confidence_intervals = diff_forecast.conf_int()
print(confidence_intervals.columns)  # Inspect the column names
dates = mean_forecast.index
lower_limit = confidence_intervals.iloc[:, 0]
upper_limit = confidence_intervals.iloc[:, 1]

# Plot the forecasted data
plt.figure(figsize=(10,6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Observed (Test Set)')
plt.plot(mean_forecast.index, mean_forecast, color='r', label='Forecast')
plt.plot(dates, mean_forecast, color='r', label='Predicted')
plt.fill_between(lower_limit.index, lower_limit, upper_limit, color='pink')
plt.title('Training Forecast Compared with Test Data')
plt.xlabel('Date')
plt.ylabel('Revenue (Millions)')
plt.legend()
plt.show()

print(mean_forecast.iloc[-1])
print(confidence_intervals.iloc[-1])

# Fit ARIMA model on test data
model_test = ARIMA(test, order=best_pdq_test)
results_test = model_test.fit()
print(results_test.summary())

# Calculate mean absolute error for test data
mae_test = np.mean(np.abs(results_test.resid))
print('Mean absolute error for test data:', mae_test)

# Calculate MAE for the forecast created by the training data on the test data
forecast_test = best_model_train.get_forecast(steps=len(test))
mean_forecast_test = forecast_test.predicted_mean
mae_forecast_test = np.mean(np.abs(mean_forecast_test - test['REVENUE']))
print('Mean absolute error for training forecast on test data:', mae_forecast_test)


# Generate predictions and confidence intervals for test data
prediction = results_test.get_prediction(start=-146, end=len(test) + 90)  # Extend forecast by 90 days
mean_prediction = prediction.predicted_mean
confidence_intervals = prediction.conf_int()
lower_limit = confidence_intervals.iloc[:, 0]
upper_limit = confidence_intervals.iloc[:, 1]




# Calculate the 90-day moving average for the test set and extend it by an additional 90 days using the ARIMA model
extended_test = pd.concat([train['REVENUE'], test['REVENUE'], mean_prediction[-90:]])
moving_average_90_extended = extended_test.rolling(window=90).mean()

# Plot the training data, test data, and the 90-day moving average with its confidence interval
plt.figure(figsize=(12, 4))
plt.plot(train.index, train['REVENUE'], label='Training Data')
plt.plot(test.index, test['REVENUE'], label='Observed (Test Set)')
plt.plot(moving_average_90_extended.index, moving_average_90_extended, color='b', label='90-Day Moving Average')
plt.title('Training Data, Test Data, and 90-Day Moving Average with 90-Day Extension')
plt.xlabel('Date')
plt.ylabel('Revenue (Millions)')
plt.legend()
plt.grid(True)
plt.show()


# Extend the moving average by an additional 90 days
extended_moving_average = pd.concat([revData['REVENUE'], mean_prediction[-90:]]).rolling(window=90).mean()

# Plot the moving average for the entire dataset with the 90-day forecast and confidence interval
plt.figure(figsize=(12, 4))
plt.plot(revData.index, revData['REVENUE'], label='Original Time Series')
plt.plot(extended_moving_average.index, extended_moving_average, color='r', label='Moving Average')
plt.plot(mean_prediction.index[-90:], mean_prediction[-90:], color='b', label='90-Day Forecast')
plt.fill_between(lower_limit.index[-90:], lower_limit[-90:], upper_limit[-90:], color='pink', alpha=0.3, label='Confidence Interval')
plt.title('Moving Average for the Entire Dataset with 90-Day Forecast')
plt.xlabel('Date')
plt.ylabel('Revenue (Millions)')
plt.legend()
plt.grid(True)
plt.show()


# Plot the final 90 days of the test data, the 90-day forecast with the confidence interval, and the 90-day moving average
plt.figure(figsize=(12, 4))
plt.plot(test.index[-90:], test['REVENUE'][-90:], label='Observed (Test Set)')
plt.plot(mean_prediction.index[-90:], mean_prediction[-90:], color='r', label='Forecast')
plt.fill_between(lower_limit.index[-90:], lower_limit[-90:], upper_limit[-90:], color='pink')
plt.plot(extended_moving_average.index[-180:], extended_moving_average[-180:], color='b', label='90-Day Moving Average')
plt.title('Final 90 Days of Test Data, 90-Day Forecast, and 90-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Revenue (Millions)')
plt.legend()
plt.grid(True)
plt.show()