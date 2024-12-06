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

# 对数转换
data['log_Population'] = np.log(data['Population'])


# 1.1 数据收集和清理
# Fill missing values 三次插值法
def interpolate_missing(series):
    non_null = series.notnull()
    indices = np.arange(len(series))
    interp = interpolate.interp1d(indices[non_null], series[non_null], kind='cubic', fill_value='extrapolate')
    return pd.Series(interp(indices), index=series.index)


for column in data.columns:
    if data[column].isnull().any():
        data[column] = interpolate_missing(data[column])

# Check for remaining missing values
print("处理后的缺失值统计: ")
print(data.isnull().sum())

# 2. 描述性统计
# Basic statistical summary
print("数据描述: ")
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


# 中位数代替异常值
def replace_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    for col in df.columns:
        df[col] = df[col].where((df[col] >= lower_bound[col]) & (df[col] <= upper_bound[col]), df[col].median())
    return df


# 处理异常值
data = replace_outliers_iqr(data)
print("替换后的数据：")
print(data)

# 绘制时间序列图,观察趋势和季节性
# Time series plot for log_Population
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['log_Population'])
plt.title('China log_Population Over Time')
plt.xlabel('Year')
plt.ylabel('log_Population')
plt.show()

# 时间序列绘图，添加移动平均线、趋势线
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['log_Population'], label='log_Population')
# 添加滚动均线
data['Population_MA'] = data['log_Population'].rolling(window=3).mean()
plt.plot(data.index, data['Population_MA'], label='3-Year Moving Average', linestyle='--')
plt.title('China log_Population Over Time with Moving Average')
plt.xlabel('Year')
plt.ylabel('log_Population')
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
- 对目标变量(log_Population)进行ADF检验
- 如果非平稳,进行差分直到达到平稳

2.2 季节性检验 (咱数据以年为单位，不需要季节性检验)
- 使用季节性分解方法检测季节性模式
"""


# 2.1 平稳性检验
# Stationary test for log_Population
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')


print("ADF Test for log_Population:")
adf_test(data['log_Population'])

# If non-stationary, difference the series
if adfuller(data['log_Population'])[1] > 0.05:
    data['Population_diff'] = data['log_Population'].diff()
    print("\nADF Test for Differenced log_Population:")
    adf_test(data['Population_diff'].dropna())

# 2.2 季节性检验
# from statsmodels.tsa.seasonal import seasonal_decompose
#
# # 季节性分解
# result = seasonal_decompose(data['log_Population'], model='', period=1)
# result.plot()
# plt.show()

"""
## 3. 特征工程和选择

3.1 滞后项创建
- 为每个解释变量创建多个滞后项(如lag1, lag2, lag3)
- 使用VIF（方差膨胀因子）来检测多重共线性，并移除VIF值过高的变量
"""

# 3.1 滞后项创建
# Feature engineering: create lag features
for col in data.columns:
    if col != 'log_Population':
        for lag in range(1, 4):  # Creating lags 1, 2, and 3
            data[f'{col}_lag{lag}'] = data[col].shift(lag)

# Remove rows with NaN values after creating lag features
data = data.dropna()

# Prepare features (X) and target (y)
X = data.drop('log_Population', axis=1)
y = data['log_Population']

# 标准化特征 (Standardize the features)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


# 计算VIF值 (Calculating VIF to detect curvilinearity)
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


# Removing multi collinear features
print("Remove multi collinear features")
while True:
    vif = calculate_vif(X_scaled)
    if vif['VIF'].max() > 10:
        removed_var = vif.loc[vif['VIF'].idxmax(), 'Variable']
        print(f"Dropping column {removed_var} with VIF: {vif['VIF'].max()}")
        X_scaled = X_scaled.drop(columns=removed_var)
    else:
        break

print("\nFinal VIF values after removing multi collinear features:")
print(calculate_vif(X_scaled))

"""
## 4. 模型构建和评估

4.1 ARIMA模型
- 仅使用目标变量(log_Population)构建ARIMA模型
- 确定最佳的p, d, q参数

4.2 ARIMAX模型
- 将选定的外生变量纳入ARIMAX模型
- 比较ARIMA和ARIMAX的预测性能

4.3 模型诊断
- 检查残差的白噪声特性
- 进行模型假设检验(如正态性、同方差性)

4.4 特征选择与改进
- 使用Lasso回归进行特征选择, 观察特征筛选后的ARIMAX模型是否表现更好
- 考虑使用PCA降维,但需要权衡解释性
"""
test_length = 5
# Prepare training and testing features and target
X_train_scaled = X_scaled.iloc[:-test_length]  # Training features
y_train = y.iloc[:-test_length]  # Training target
X_test_scaled = X_scaled.iloc[-test_length:]  # Testing features
y_test = y.iloc[-test_length:]  # Testing target

# 4.1 ARIMA 模型（不包含外生变量）
print("\n-----------------ARIMA Model-----------------------------")
arima_model = auto_arima(y_train, seasonal=False, trace=True)
print(arima_model.summary())

arima_forecast = arima_model.predict(n_periods=test_length)
# 将log_Population恢复到原始尺度
arima_forecast_exp = np.exp(arima_forecast)
y_test_exp = np.exp(y_test)

mse = mean_squared_error(y_test_exp, arima_forecast_exp)
mae = mean_absolute_error(y_test_exp, arima_forecast_exp)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# 4.2 ARIMAX模型
print("\n-----------------SARIMA Model-----------------------------")
# Build ARIMAX model using training data
model = auto_arima(y_train, exogenous=X_train_scaled, start_p=1, start_q=1, max_p=5, max_q=5, m=1,
                   start_P=0, seasonal=True, d=2, D=1, trace=True, error_action='ignore',
                   suppress_warnings=True, stepwise=False, n_fits=50, grid_search=True)

print(model.summary())

# Forecast on the test set
arimax_forecast = model.predict(n_periods=test_length, exogenous=X_test_scaled)
# 将log_Population恢复到原始尺度
arimax_forecast_exp = np.exp(arimax_forecast)

mse = mean_squared_error(y_test_exp, arimax_forecast_exp)
mae = mean_absolute_error(y_test_exp, arimax_forecast_exp)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, y, label='Observed (Real Values)', color='blue')
plt.plot(y_test.index, arimax_forecast, label='Forecasted Values', color='red', linestyle='--')
plt.title('log_Population Forecast vs Actual (Test Set)')
plt.xlabel('Year')
plt.ylabel('log_Population')
plt.legend()
plt.show()

# Residual Analysis
residuals = pd.DataFrame(model.resid())
residuals.plot(kind='kde')
plt.title('Residual Density Plot')
plt.show()

# ACF and PACF plot for residuals
# 获取样本量
sample_size = len(residuals)

# 设置nlags为样本量的50%，并且取整
nlags = min(sample_size // 2, 13)  # 设置一个安全的滞后值，例如13
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(residuals)
plot_pacf(residuals, method='ywm', lags=nlags)
plt.show()

# Q-Q plot
from scipy import stats

stats.probplot(residuals.squeeze(), dist="norm", plot=plt)
plt.title("Q-Q plot for residual")
plt.show()

# 4.4 特征选择与改进
print("\n-----------------Lasso Model-----------------------------")
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

# 准备用于Lasso的数据
X_lasso = X_scaled.values
y_lasso = y.values

# 使用时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 使用LassoCV进行特征选择
lasso_cv = LassoCV(cv=tscv, max_iter=1000, random_state=42)
lasso_cv.fit(X_lasso, y_lasso)

# 获取选定的特征
selected_features = X_scaled.columns[lasso_cv.coef_ != 0]

print("\nSelected features by Lasso:")
print(selected_features)

# 使用选定的特征更新X_scaled
X_scaled_selected = X_scaled[selected_features]

print("\nSelected features and their coefficients:")
# 打印每个选定特征的系数
for feature, coef in zip(selected_features, lasso_cv.coef_[lasso_cv.coef_ != 0]):
    print(f"{feature}: {coef}")

# 更新训练集和测试集
X_train_scaled = X_scaled_selected.iloc[:-test_length]
X_test_scaled = X_scaled_selected.iloc[-test_length:]

print("\n")
# 重新构建ARIMAX模型使用选定的特征
model_selected = auto_arima(y_train, exogenous=X_train_scaled, start_p=1, start_q=1, max_p=5, max_q=5, m=1,
                            start_P=0, seasonal=True, d=2, D=1, trace=True, error_action='ignore',
                            suppress_warnings=True, stepwise=False, n_fits=50, grid_search=True)

print(model_selected.summary())

# 使用选定特征的模型进行预测
forecast_selected = model_selected.predict(n_periods=test_length, exogenous=X_test_scaled)
# 将log_Population恢复到原始尺度
forecast_selected_exp = np.exp(forecast_selected)

# 计算新模型的误差
mse_selected = mean_squared_error(y_test_exp, forecast_selected_exp)
mae_selected = mean_absolute_error(y_test_exp, forecast_selected_exp)

print(f'Mean Squared Error (Selected Features): {mse_selected}')
print(f'Mean Absolute Error (Selected Features): {mae_selected}')

# 对比原模型和使用选定特征的模型
plt.figure(figsize=(12, 6))
plt.plot(data.index, y, label='Observed (Real Values)', color='blue')
plt.plot(y_test.index, arimax_forecast, label='Forecast (All Features)', color='red', linestyle='--')
plt.plot(y_test.index, forecast_selected, label='Forecast (Selected Features)', color='green', linestyle=':')
plt.title('log_Population Forecast Comparison')
plt.xlabel('Year')
plt.ylabel('log_Population')
plt.legend()
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
