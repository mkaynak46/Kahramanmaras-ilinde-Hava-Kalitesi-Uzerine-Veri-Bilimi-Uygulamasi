# -*- coding: utf-8 -*-
"""


@author: Musty Kaynak
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("Veri Setindeki class")

#Veri setini yükleme
veri = pd.read_csv("KahramanmarasVeriset.csv")

#Sınıf sayısını belirleme
label_encoder = LabelEncoder().fit(veri.PM10)
labels = label_encoder.transform(veri.PM10)
classes = list(label_encoder.classes_)

x= veri.drop(["PM10",], axis=1)
y = labels

#verileri standartlaştırma
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)
#Eğitim ve Test verileri
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train,y_test = train_test_split(X,y,test_size = 0.2) 
#çıktı değerlerini katagorileştirme
from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#YSA oluşturmak
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16,input_dim=20,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(8,input_dim=20,activation="relu"))
model.add(Dense(4,activation="softmax"))
model.summary()

#model derlenmesi
model.compile(loss = "categorical_crossentropy",optimizer="adam",metrics = ["accuracy"])
#modelin eğitilmesi
model.fit(X_train,y_train, validation_data = (X_test,y_test) , epochs=20)
#gerekli değerleri gösterilmesi
print("Ortalama eğitim kaybı:",np.mean(model.history.history["loss"]))
print("Ortalama eğitim başarımı:",np.mean(model.history.history["accuracy"]))
print("Ortalama doğrulama kaybı:",np.mean(model.history.history["val_loss"]))
print("Ortalama doğrulama başarımı:",np.mean(model.history.history["val_accuracy"]))
#eğitim ve doğrulama başarım gösterimi
import matplotlib.pyplot as plt
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model başarımı")
plt.ylabel("Başarım")
plt.xlabel("Epok")
plt.legend(["Eğitim","Test"] , loc="upper left")
plt.show()

plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model kaybı")
plt.ylabel("Kayıp")
plt.xlabel("Epok")
plt.legend(["Eğitim","Test"] , loc="upper left")
plt.show()

