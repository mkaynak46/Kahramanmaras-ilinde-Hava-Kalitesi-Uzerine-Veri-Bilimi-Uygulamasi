
# Tensorflow'u Kurma
# pip install tensorflow

# Keras'ı Yükleme
# pip install - upgrade keras

# Adım 1 - Veri Önişleme

# Kütüphaneleri içe aktarma
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

script_dir = os.path.dirname(__file__)
abs_file_path = os.path.join(script_dir, 'Kahramanmaras.csv')

# Veri kümesini içe aktarma
dataset = pd.read_csv(abs_file_path)
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Kategorik verileri kodlama
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Veri kümesini Eğitim kümesine ve Test kümesine bölme
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Özellik Ölçeklendirme
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Adım 2 -  YSA 

# Keras kütüphanelerini ve paketlerini içe aktarma
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras import backend

# YSA'nın başlatılması
classifier = Sequential()

# Giriş katmanını ve ilk gizli katmanı ekleme
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# İkinci gizli katmanı ekleme
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Çıktı katmanı ekleme
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# YSA'nın derlenmesi
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# YSA'nın Eğitim setine yerleştirilmesi
classifier.fit(X_train, y_train, batch_size=10, epochs=100, validation_split=0.1)

# Adım 3 - Tahmin yapma ve modeli değerlendirme

# Test seti sonuçlarını tahmin etme
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Karışıklık Matrisini Oluşturma
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
backend.clear_session()
