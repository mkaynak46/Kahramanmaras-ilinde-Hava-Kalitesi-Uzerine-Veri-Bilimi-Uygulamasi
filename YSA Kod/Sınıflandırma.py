# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:16:47 2020

@author: Musty Kaynak
"""

# Sınıflandırma şablonu

# Kütüphaneleri içe aktarma
import numpy as np
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath("__file__"))
abs_file_path = os.path.join(script_dir, 'Kahramanmaras.csv')

# Veri kümesini içe aktarma
dataset = pd.read_csv(abs_file_path)
X = dataset.iloc[:, [0, 2]].values
y = dataset.iloc[:, 4].values

# Veri kümesini Eğitim kümesine ve Test kümesine bölme
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Özellik Ölçeklendirme
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Sınıflandırıcıyı Eğitim setine yerleştirme
# Sınıflandırıcıyı burada oluşturun
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Test seti sonuçlarını tahmin etme
y_pred = classifier.predict(X_test)

# Karışıklık Matrisini Oluşturma
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Eğitim seti sonuçlarını görselleştirme
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Sınıflandırıcı (Eğitim seti)')
plt.xlabel('SO2')
plt.ylabel('PM10')
plt.legend()
plt.show()

# Test seti sonuçlarını görselleştirme
from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Sınıflandırıcı (Eğitim seti)')
plt.xlabel('SO2')
plt.ylabel('PM10')
plt.legend()
plt.show()