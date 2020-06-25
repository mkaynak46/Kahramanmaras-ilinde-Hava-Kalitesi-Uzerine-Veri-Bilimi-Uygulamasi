import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_excel("KahramanmarasVeriset.xlsx")

df = pd.DataFrame(data)
print(df)

#ilk 5 verinin gösterilmesi
data.head()

plt.figure(figsize=(10,10))
plt.title("Türkiye'nin Enflasyon Oranı")
plt.xlabel("Tarih")

plt.bar(data.Tarih,data.PM10)

plt.show()