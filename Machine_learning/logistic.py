import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 데이터 불러오기
data = pd.read_csv("./data/iris.csv")

# Feature와 Target 설정
x = data[['petal.length', 'sepal.length', 'petal.width', 'sepal.width']]  # Feature
y = data['variety']  # Target: 꽃의 종

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 생성
model = LogisticRegression()

# 모델 훈련
model.fit(x_train, y_train)

# 예측
y_pred = model.predict(x_test)

# 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
