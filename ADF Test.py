import pandas as pd
import matplotlib.pyplot as plt

# Load data from Excel files
data = pd.read_excel("Data-Integrate.xlsx", usecols='A:L', header=3, nrows=39)

data.set_index('Year', inplace=True)
data.index = pd.to_datetime(data.index, format='%Y') + pd.offsets.YearEnd(0)
data = data.asfreq('A-DEC')

# Differencing to make the data stationary
data['Population_diff'] = data['Population'].diff().dropna()
plt.figure(figsize=(12, 6))

plt.plot(data['Population_diff'], label='Differenced Population')
plt.title('First-order Differenced Population Series')
plt.xlabel('Year')
plt.ylabel('Population Difference')
plt.legend()
plt.show()

"""
ADF Stationary Test - First Differencing
"""
print("\n----------------First-order differencing Stationary Test----------------")

from statsmodels.tsa.stattools import adfuller

# ADF test
adf_result = adfuller(data['Population_diff'].dropna())

print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print(f"Critical Values: {adf_result[4]}")

# Interpret the result
if adf_result[1] < 0.05:
    print("Reject the null hypothesis (H0): The data is stationary.")
else:
    print("Fail to reject the null hypothesis (H0): The data is non-stationary.")

# Visualizing ADF test results
adf_statistic = adf_result[0]
critical_values = adf_result[4]

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [adf_statistic, adf_statistic], label=f'ADF Statistic: {adf_statistic:.2f}', color='blue')
plt.plot([0, 1], [critical_values['1%'], critical_values['1%']], label='Critical Value (1%)', linestyle='--', color='red')
plt.plot([0, 1], [critical_values['5%'], critical_values['5%']], label='Critical Value (5%)', linestyle='--', color='green')
plt.plot([0, 1], [critical_values['10%'], critical_values['10%']], label='Critical Value (10%)', linestyle='--', color='orange')
plt.legend()
plt.title('ADF Test for First-order differencing: ADF Statistic vs. Critical Values')
plt.show()

"""
Second-order differencing
"""
print("\n----------------Second-order differencing Stationary Test----------------")

# Performing second-order differencing
data['Population_diff2'] = data['Population_diff'].diff().dropna()

# Plotting the second differenced series
plt.figure(figsize=(12, 6))
plt.plot(data['Population_diff2'], label='Second-order Differenced Population')
plt.title('Second-order Differenced Population Series')
plt.xlabel('Year')
plt.ylabel('Population Difference (Second-order)')
plt.legend()
plt.show()

# ADF test on the second differenced series
adf_result_2 = adfuller(data['Population_diff2'].dropna())

print(f"ADF Statistic: {adf_result_2[0]}")
print(f"p-value: {adf_result_2[1]}")
print(f"Critical Values: {adf_result_2[4]}")

if adf_result_2[1] < 0.05:
    print("Reject the null hypothesis (H0): The data is stationary.")
else:
    print("Fail to reject the null hypothesis (H0): The data is non-stationary.")

# Visualizing ADF test results
adf_statistic_2 = adf_result_2[0]
critical_values_2 = adf_result_2[4]

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [adf_statistic_2, adf_statistic_2], label=f'ADF Statistic: {adf_statistic_2:.2f}', color='blue')
plt.plot([0, 1], [critical_values_2['1%'], critical_values_2['1%']], label='Critical Value (1%)', linestyle='--', color='red')
plt.plot([0, 1], [critical_values_2['5%'], critical_values_2['5%']], label='Critical Value (5%)', linestyle='--', color='green')
plt.plot([0, 1], [critical_values_2['10%'], critical_values_2['10%']], label='Critical Value (10%)', linestyle='--', color='orange')
plt.legend()
plt.title('ADF Test for Second-order differencing: ADF Statistic vs. Critical Values')
plt.show()

# The series after second differencing is stationary, which means we can use ARIMA model to forecast our series
