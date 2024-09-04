import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np

file_path = r'D:\Desktop\Python\ML model\dataset.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name)

sampled_df = df.sample(n=50, random_state=42)

feature_columns = ['Test_Time(s)', 'Step_Time(s)', 'Step_Index', 'Voltage(V)', 'Current(A)']
target_column = 'Surface_Temp(degC)'

X = sampled_df[feature_columns]
y = sampled_df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print('Feature Importances:', model.feature_importances_)

plot_df = X_test.copy()
plot_df[target_column] = y_test
plot_df['Predicted'] = y_pred
plot_df['Test_Time(s)'] = df.loc[X_test.index, 'Test_Time(s)']
plot_df.sort_values('Test_Time(s)', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(plot_df['Test_Time(s)'], plot_df[target_column], color='blue', linestyle='-', marker='o', label='Actual Temperature')
plt.plot(plot_df['Test_Time(s)'], plot_df['Predicted'], color='green', linestyle='--', marker='x', label='Predicted Temperature')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs. Predicted Temperature Over Time using Decision Tree Regressor')
plt.grid(True)
plt.legend()
plt.show()

temperature_threshold = 75 
anomaly_model = IsolationForest(contamination=0.01, random_state=42)
anomaly_model.fit(X[['Surface_Temp(degC)']])
anomalies = anomaly_model.predict(X[['Surface_Temp(degC)']])
risk_indices = np.where(y_pred > temperature_threshold)[0]

plt.figure(figsize=(12, 6))
plt.scatter(X_test['Test_Time(s)'].iloc[risk_indices], y_test.iloc[risk_indices], color='red', label='Risk of Overheating')
plt.plot(plot_df['Test_Time(s)'], plot_df[target_column], color='blue', linestyle='-', marker='o', label='Actual Temperature')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature with Overheating Risk Points')
plt.grid(True)
plt.legend()
plt.show()
