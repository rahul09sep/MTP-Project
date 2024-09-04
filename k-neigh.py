import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel(r'D:\Desktop\Python\ML model\dataset.xlsx')

# df = df.head(55550)
sampled_df = df.sample(n=50, random_state=42)

X = sampled_df[['Test_Time(s)', 'Step_Time(s)', 'Step_Index', 'Voltage(V)', 'Current(A)']]
y = sampled_df['Surface_Temp(degC)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Root Mean Squared Error: {rmse}')

plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, color='black', label='Ideal Prediction Line')

plt.plot(y_test, y_pred, color='red', label='Prediction Line')

plt.xlabel('Actual Surface Temp (degC)')
plt.ylabel('Predicted Surface Temp (degC)')
plt.title('Actual vs Predicted Surface Temperature')
plt.legend()
plt.grid(True)
plt.show()
