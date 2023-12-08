import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
data = pd.read_csv("./data/iris.csv")

# Feature와 Target 설정
x = data[['sepal.length','sepal.width','petal.length']]  # Feature: 꽃잎 길이, 꽃잎 너비, 꽃받침 길이
y = data['petal.width']  # Target: 꽃받침 너비

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 훈련
model.fit(x_train, y_train)

# 예측
y_pred = model.predict(x_test)

# 모델 성능 평가
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
