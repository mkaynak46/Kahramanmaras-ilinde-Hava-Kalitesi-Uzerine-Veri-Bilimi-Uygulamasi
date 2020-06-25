# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:12:57 2020

@author: Musty Kaynak
"""


import pandas as pd
from bokeh.plotting import figure, output_file, show
data=pd.read_excel("KahramanmarasVeriset.xlsx")

df = pd.DataFrame(data)
print(df)


x = data.Tarih
y = data.SO2
# html olarak açılacak çıktının adı
output_file("So2cizgi.html") 
p = figure(title="SO2 Değişim Çizgi Grafiği", x_axis_label='x', y_axis_label='y') 
# Cizgi olarak x ve y eksenlerini ciz 
# legend adı: Sicaklik 
# Çizgi kalınlığı: 2 
p.line(x, y, legend_label="SO2.", line_width=2) 
# Çizimi göster 
show(p)