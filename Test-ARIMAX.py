import pandas as pd
from pmdarima import auto_arima
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data from Excel files
data = pd.read_excel("Data-Integrate.xlsx", usecols='A:L', header=3, nrows=39)

data.set_index('Year', inplace=True)
data.index = pd.to_datetime(data.index, format='%Y') + pd.offsets.YearEnd(0)
data = data.asfreq('A-DEC')

# Fill missing values
data.fillna(method='ffill', inplace=True)

# 创建多个滞后项，比如滞后1, 滞后2, 滞后3
lag_values = [1, 2, 3]


# 创建滞后项函数
def create_lags(data, col, lags):
    for lag in lags:
        data[f'{col}_lag{lag}'] = data[col].shift(lag)
    return data


# 为每个外生变量创建滞后项
exogenous_vars = data.columns
for var in exogenous_vars:
    data = create_lags(data, var, lag_values)

# 删除包含 NaN 的行（因为滞后项会引入 NaN）
data.dropna(inplace=True)

# 划分训练集和测试集
train = data.iloc[:-5]  # Use all but the last 5 years for training
test = data.iloc[-5:]  # Use the last 5 years for testing

# 定义用于 ARIMAX 的外生变量（包括所有滞后项）
exog_cols = [f'{var}_lag{lag}' for var in exogenous_vars for lag in lag_values]
exog_train = train[exog_cols]
exog_test = test[exog_cols]


# 自动去除高共线性变量的函数
def remove_high_vif(exog_data, vif_threshold=5):
    vif_data = pd.DataFrame()
    vif_data["feature"] = exog_data.columns
    vif_data["VIF"] = [variance_inflation_factor(exog_data.values, i) for i in range(len(exog_data.columns))]

    # 不断去除VIF最高的变量，直到所有变量的VIF都低于阈值
    while vif_data['VIF'].max() > vif_threshold:
        highest_vif = vif_data.loc[vif_data['VIF'].idxmax()]
        print(f"Removing {highest_vif['feature']} with VIF: {highest_vif['VIF']}")
        exog_data = exog_data.drop(columns=[highest_vif['feature']])

        # 重新计算VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = exog_data.columns
        vif_data["VIF"] = [variance_inflation_factor(exog_data.values, i) for i in range(len(exog_data.columns))]

    return exog_data, vif_data


# 去除共线性高的变量
exog_train_reduced, final_vif = remove_high_vif(exog_train)
print(final_vif)
exog_test = exog_test[final_vif['feature']]

# # Initialize the scaler
# scaler = StandardScaler()
#
# # Fit and transform the exogenous variables
# exog_train_scaled = scaler.fit_transform(exog_train)
# exog_test_scaled = scaler.transform(exog_test)

# 定义目标变量
y_train = train['Population']
y_test = test['Population']

# 使用 auto_arima 选择最佳的 ARIMAX 参数
model = auto_arima(y_train,
                   exogenous=exog_train_reduced,
                   seasonal=False,
                   stepwise=True,
                   suppress_warnings=True)

# 打印选择的 ARIMAX 模型参数
print(model.summary())

# 在测试集上进行预测
y_pred = model.predict(n_periods=len(y_test), exogenous=exog_test)

# 输出结果
print("Predictions:", y_pred)

# 评估模型表现（MSE, MAE等指标）
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE: {mse}')
print(f'MAE: {mae}\n')
