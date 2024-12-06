import pandas as pd
from pmdarima import auto_arima
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy import interpolate

# Load data from Excel files
data = pd.read_excel("Data-Integrate.xlsx", usecols='A:L', header=3, nrows=39)

"""
数据准备和初步分析
1.1 数据收集和清理
- 确保所有变量的时间序列完整性
- 处理缺失值和异常值

1.2 描述性统计
- 计算每个变量的基本统计量
- 绘制时间序列图,观察趋势和季节性

1.3 相关性分析
- 计算变量间的相关系数矩阵
- 绘制热力图可视化相关性
"""

data.set_index('Year', inplace=True)
data.index = pd.to_datetime(data.index, format='%Y') + pd.offsets.YearEnd(0)
data = data.asfreq('A-DEC')


# 1.1 数据收集和清理
# Fill missing values
def interpolate_missing(series):
    non_null = series.notnull()
    indices = np.arange(len(series))
    interp = interpolate.interp1d(indices[non_null], series[non_null], kind='cubic', fill_value='extrapolate')
    return pd.Series(interp(indices), index=series.index)


for column in data.columns:
    if data[column].isnull().any():
        data[column] = interpolate_missing(data[column])

# Check for remaining missing values
print(data.isnull().sum())

# 2. 描述性统计
# Basic statistical summary
print(data.describe())


# 基于 IQR 的异常值检测
def detect_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    return outliers


# 查看异常值
outliers = detect_outliers_iqr(data)
print(f"异常值检测结果：\n{outliers.sum()}")

# 绘制时间序列图,观察趋势和季节性
# Time series plot for Population
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Population'])
plt.title('China Population Over Time')
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()

# 时间序列绘图，添加移动平均线、趋势线
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Population'], label='Population')
# 添加滚动均线
data['Population_MA'] = data['Population'].rolling(window=3).mean()
plt.plot(data.index, data['Population_MA'], label='3-Year Moving Average', linestyle='--')
plt.title('China Population Over Time with Moving Average')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.show()

# 1.3 相关性分析
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

"""
## 2. 时间序列特性检验

2.1 平稳性检验
- 对目标变量(Population)进行ADF检验
- 如果非平稳,进行差分直到达到平稳

2.2 季节性检验
- 使用季节性分解方法检测季节性模式
"""


# 2.1 平稳性检验
# Stationary test for Population
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')


print("ADF Test for Population:")
adf_test(data['Population'])

# If non-stationary, difference the series
if adfuller(data['Population'])[1] > 0.05:
    data['Population_diff'] = data['Population'].diff()
    print("\nADF Test for Differenced Population:")
    adf_test(data['Population_diff'].dropna())

# 2.2 季节性检验


"""
## 3. 特征工程和选择

3.1 滞后项创建
- 为每个解释变量创建多个滞后项(如lag1, lag2, lag3)
- 使用VIF（方差膨胀因子）来检测多重共线性，并移除VIF值过高的变量

3.2 特征选择
- 使用Lasso回归进行特征选择
- 考虑使用PCA降维,但需要权衡解释性
"""

# 3.1 滞后项创建
# Feature engineering: create lag features
for col in data.columns:
    if col != 'Population':
        for lag in range(1, 4):  # Creating lags 1, 2, and 3
            data[f'{col}_lag{lag}'] = data[col].shift(lag)

# Remove rows with NaN values after creating lag features
data = data.dropna()

# Prepare features and target
X = data.drop('Population', axis=1)
y = data['Population']

# Standardize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


# Check for curvilinearity
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


while True:
    vif = calculate_vif(X_scaled)
    if vif['VIF'].max() > 10:
        removed_var = vif.loc[vif['VIF'].idxmax(), 'Variable']
        print(f"Dropping column {removed_var} with VIF: {vif['VIF'].max()}")
        X_scaled = X_scaled.drop(columns=removed_var)
    else:
        break

print("Final VIF values after removing multi collinear features:")
print(calculate_vif(X_scaled))

# 3.2 特征选择


"""
## 4. 模型构建和评估

4.1 ARIMA模型
- 仅使用目标变量(Population)构建ARIMA模型
- 确定最佳的p, d, q参数

4.2 ARIMAX模型
- 将选定的外生变量纳入ARIMAX模型
- 比较ARIMA和ARIMAX的预测性能

4.3 模型诊断
- 检查残差的白噪声特性
- 进行模型假设检验(如正态性、同方差性)
"""

# ARIMAX model
# Note: You may need to adjust the order and seasonal_order parameters
model = auto_arima(y, exogenous=X_scaled, start_p=1, start_q=1, max_p=5, max_q=5, m=1,
                   start_P=0, seasonal=True, d=1, D=1, trace=True, error_action='ignore',
                   suppress_warnings=True, stepwise=False, n_fits=50, grid_search=True)

print(model.summary())

# Forecast
n_periods = 5
forecast = model.predict(n_periods=n_periods, exogenous=X_scaled.iloc[-n_periods:])

# Error evaluation
y_true = y[-n_periods:]
print("y_true\n", y_true)
mse = mean_squared_error(y_true, forecast)
mae = mean_absolute_error(y_true, forecast)

print("Forecast for the next 5 years:")
print(forecast)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, y, label='Observed (Real Values)', color='blue')
forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(years=1), periods=n_periods, freq='A-DEC')
plt.plot(forecast_index, forecast, label='Forecasted Values', color='red', linestyle='--')
plt.title('Population Forecast')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.show()

# Residual Analysis
residuals = pd.DataFrame(model.resid())
residuals.plot(kind='kde')
plt.title('Residual Density Plot')
plt.show()

# ACF and PACF plot for residuals
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(residuals)
plot_pacf(residuals, method='ywm')
plt.show()

# Q-Q plot
from scipy import stats

stats.probplot(residuals.squeeze(), dist="norm", plot=plt)
plt.show()

"""
## 5. 预测和结果分析

5.1 生成预测
- 使用最佳模型进行短期和长期预测

5.2 结果解释
- 分析各变量对人口预测的影响
- 比较预期效果和实际结果
"""

"""
## 6. 结论和推广

6.1 总结发现
- 概括关键发现和新的洞察

6.2 方法论推广
- 讨论如何将此分析框架应用于其他社会经济指标
"""