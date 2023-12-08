import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv("./data/iris.csv")

# 데이터 전처리
x = data.iloc[:, [0, 1, 2, 3]].values
y = data['variety'].replace(['Setosa', 'Virginica', 'Versicolor'], [1, 2, 0]).values

# 엘보우 메서드
wcss = []  # WCSS 값을 저장할 리스트
for i in range(1, 11):  # 클러스터 수를 1에서 10까지 시도
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)  # x는 데이터
    wcss.append(kmeans.inertia_)  # WCSS 값을 리스트에 추가

# WCSS 값 그래프를 그림
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# 실루엣 스코어
silhouette_scores = []  # 실루엣 스코어를 저장할 리스트
for i in range(2, 11):  # 클러스터 수를 2에서 10까지 시도
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)  # x는 데이터
    silhouette_scores.append(silhouette_score(x, kmeans.labels_))

# 실루엣 스코어 그래프를 그림
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')

# 클러스터링 수행
km = KMeans(n_clusters=3)  # 적절한 클러스터 수 선택 (예: 3)
km.fit(x)

y_pre = km.labels_

sc = (y == y_pre).sum()
acc = sc / len(y)
print(f"Accuracy: {acc}")

# 클러스터링 결과 시각화 (예: 2D 그래프)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x[:, 0], x[:, 1], c=y_pre, cmap='viridis')
plt.title('Clustering Result')
plt.xlabel('sepal.length')
plt.ylabel('sepal.width')

plt.subplot(1, 2, 2)
plt.scatter(x[:, 2], x[:, 3], c=y_pre, cmap='viridis')
plt.title('Clustering Result')
plt.xlabel('petal.length')
plt.ylabel('petal.width')

plt.show()
