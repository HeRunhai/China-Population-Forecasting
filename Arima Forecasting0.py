import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pmdarima import auto_arima
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# Load data from Excel files
birth_rate = pd.read_excel("BirthRate-China.xlsx", usecols='A:B').iloc[::-1].reset_index(drop=True)
death_rate = pd.read_excel("DeathRate-China.xlsx", usecols='A:B').iloc[::-1].reset_index(drop=True)
gdp = pd.read_excel("GDP-China.xlsx", sheet_name='Data', header=None, names=["Year", "GDP"], usecols='B:C', skiprows=5, nrows=39)
population = pd.read_excel("Population-China.xlsx", sheet_name='Data', header=None, names=["Year", "Population"], usecols='B:C', skiprows=10, nrows=39)
urban_rate = pd.read_excel("UrbanRate-China.xlsx", header=None, sheet_name='Data', names=["Year", "Urban Rate"], usecols='B:C', skiprows=5, nrows=39)

# Merge all data into one DataFrame based on the 'Year' column
data = pd.merge(population, birth_rate, on="Year", how="left")
data = pd.merge(data, death_rate, on="Year", how="left")
data = pd.merge(data, gdp, on="Year", how="left")
data = pd.merge(data, urban_rate, on="Year", how="left")

# Select relevant columns
data = data[['Year', 'Population', 'Birth Rate', 'Death Rate', 'GDP', 'Urban Rate']]
data.set_index('Year', inplace=True)
data.index = pd.to_datetime(data.index, format='%Y') + pd.offsets.YearEnd(0)
data = data.asfreq('A-DEC')

# Fill missing values
data.fillna(method='ffill', inplace=True)

# 虽然外生变量与人口相关，但从实际经济学角度看，GDP、出生率、死亡率等变量可能对人口的影响并不是即时的，而是具有一定滞后性。
# 例如，GDP 增长对人口的影响可能会在几年后显现，可以尝试为外生变量引入滞后项（lag），如 GDP_t-1, BirthRate_t-1，这在预测任务中有时能有效提升模型表现。
# Introduce lagged variables (lag 1), 测试得知，lag1效果最好
data['GDP_lag1'] = data['GDP'].shift(1)
data['Birth Rate_lag1'] = data['Birth Rate'].shift(1)
data['Death Rate_lag1'] = data['Death Rate'].shift(1)
data['Urban Rate_lag1'] = data['Urban Rate'].shift(1)
data.dropna(inplace=True)  # Remove NaN values after lagging

# Data visualization
plt.figure(figsize=(12, 6))
plt.plot(data['Population'], label='Population')
plt.title('China Population Over the Years')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.show()

# Differencing to make the data stationary
data['Population_diff'] = data['Population'].diff().dropna()
plt.figure(figsize=(12, 6))
plt.plot(data['Population_diff'], label='Differenced Population')
plt.title('Differenced Population Series')
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

# ----------------Model Training and Forecasting----------------
# Splitting the data into train and test sets
train = data.iloc[:-5]  # Use all but the last 5 years for training
test = data.iloc[-5:]   # Use the last 5 years for testing

# Check correlation between the exogenous variables and the target variable (Population)
correlation_matrix = data[['Population', 'Birth Rate_lag1', 'Death Rate_lag1', 'GDP_lag1', 'Urban Rate_lag1']].corr()
# Heatmap for correlation matrix
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Correlation Matrix Heatmap')
# 旋转x轴和y轴标签
plt.xticks(rotation=45, ha='right')
# 调整图形边距
plt.tight_layout()
plt.show()

# Selecting exogenous variables with higher correlation with Population
selected_exog_vars = correlation_matrix['Population'].abs().sort_values(ascending=False)
selected_exog_vars = selected_exog_vars[selected_exog_vars > 0.5].index.tolist()
selected_exog_vars.remove('Population')

# Update exog_train and exog_test with selected variables
exog_train = train[selected_exog_vars]
exog_test = test[selected_exog_vars]

# Standardizing target variable (Population) along with exogenous variables
scaler_population = StandardScaler()

# Standardize the training and testing Population data
train_population_scaled = scaler_population.fit_transform(train[['Population']])
test_population_scaled = scaler_population.transform(test[['Population']])

# Standardizing exogenous variables
scaler_exog = StandardScaler()
exog_train_scaled = scaler_exog.fit_transform(exog_train)
exog_test_scaled = scaler_exog.transform(exog_test)

"""
ARIMA model
Choose the best ARIMA order from auto_arima
"""
print("\n----------------ARIMA Model----------------")
# ARIMA model on Population
arima_model = auto_arima(train['Population'], seasonal=False, trace=True)

# Forecasting with ARIMAX
arima_forecast = arima_model.predict(n_periods=len(test))

# Print the summary of the model
print(arima_model.summary())

# Plotting ARIMA predictions
plt.figure(figsize=(12, 6))
plt.plot(train['Population'], label='Training Data')
plt.plot(test['Population'], label='Actual Population')
plt.plot(test.index, arima_forecast, label='ARIMA Predictions', color='blue')
plt.title('ARIMA Model Predictions')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.show()

"""
ARIMAX model
"""
print("\n----------------ARIMAX Model----------------")

# ARIMAX model with auto_arima
arimax_model = auto_arima(train_population_scaled, exogenous=exog_train_scaled, seasonal=False, trace=True)

# Forecasting with ARIMAX
arimax_forecast_scaled = arimax_model.predict(n_periods=len(test), exogenous=exog_test_scaled)

# Inverse transform the ARIMAX forecast to original scale
arimax_forecast = scaler_population.inverse_transform(arimax_forecast_scaled.reshape(-1, 1))

# Print the summary of the model
print(arimax_model.summary())

# Plotting ARIMAX predictions
plt.figure(figsize=(12, 6))
plt.plot(train['Population'], label='Training Data')
plt.plot(test['Population'], label='Actual Population')
plt.plot(test.index, arimax_forecast, label='ARIMAX Predictions', color='red')
plt.title('ARIMAX Model Predictions')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.show()

"""
Lasso Regression
"""
print("\n----------------Lasso Regression Model----------------")
# Lasso Regression with GridSearchCV
lasso_params = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}  # Hyperparameters for tuning
lasso = Lasso()
grid_search = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(exog_train_scaled, train_population_scaled.ravel())

# Best parameters and performance
print("Best Lasso Alpha:", grid_search.best_params_)
best_lasso_model = grid_search.best_estimator_
lasso_forecast_scaled = best_lasso_model.predict(exog_test_scaled)

# Inverse transform the Lasso forecast to original scale
lasso_forecast = scaler_population.inverse_transform(lasso_forecast_scaled.reshape(-1, 1))

# Plotting Lasso predictions
plt.figure(figsize=(12, 6))
plt.plot(train['Population'], label='Training Data')
plt.plot(test['Population'], label='Actual Population')
plt.plot(test.index, lasso_forecast, label='Lasso Predictions', color='green')
plt.title('Lasso Model Predictions')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.show()

# ----------------Lasso Regression Feature Importance----------------
# After fitting the Lasso model, extract the coefficients (weights) to identify important features

# Get the coefficients from the best Lasso model
lasso_coefficients = pd.Series(best_lasso_model.coef_, index=selected_exog_vars)

print(lasso_coefficients.sort_values())

# Visualize the importance of features (larger absolute values indicate more important features)
plt.figure(figsize=(10, 6))
lasso_coefficients.sort_values().plot(kind='bar', color='coral')  # 更换颜色为 coral
plt.title('Feature Importance from Lasso Regression', fontsize=16)
plt.ylabel('Coefficient Value (Feature Importance)', fontsize=14)
plt.xlabel('Exogenous Features', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Model Evaluation
arima_mae = mean_absolute_error(test['Population'], arima_forecast)
arima_mse = mean_squared_error(test['Population'], arima_forecast)
arimax_mae = mean_absolute_error(test['Population'], arimax_forecast)
arimax_mse = mean_squared_error(test['Population'], arimax_forecast)
lasso_mae = mean_absolute_error(test['Population'], lasso_forecast)
lasso_mse = mean_squared_error(test['Population'], lasso_forecast)

print(f'\nARIMA Model MAE: {arima_mae}, MSE: {arima_mse}')
print(f'ARIMAX Model MAE: {arimax_mae}, MSE: {arimax_mse}')
print(f'Lasso Model MAE: {lasso_mae}, MSE: {lasso_mse}')


# Viz for MSE comparison

# 假设三个 MSE 值
mse_values = [arima_mse, arimax_mse, lasso_mse]  # 替换为你的实际MSE值
model_names = ['ARIMA', 'ARIMAX', 'Lasso']  # 替换为你要对比的模型

mse_data = pd.DataFrame({
    'Model': model_names,
    'MSE': mse_values
})

sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='MSE', data=mse_data, palette='muted')
plt.title('MSE Comparison of Different Models', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
plt.show()

