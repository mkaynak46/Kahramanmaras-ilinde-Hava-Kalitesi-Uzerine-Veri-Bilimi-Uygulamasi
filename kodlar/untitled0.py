# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:12:57 2020

@author: Musty Kaynak
"""

# Gerekli kütüphane
import pandas as pd
from bokeh.plotting import figure, output_file, show
data=pd.read_excel("KahramanmarasVeriset.xlsx")

df = pd.DataFrame(data)
print(df)

x = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
y = data.Havasicakligi

output_file("cizgi.html")  
p = figure(title="Sıcaklık Çizgi Grafiği", x_axis_label='x', y_axis_label='y') 
# Cizgi olarak x ve y eksenlerini ciz 
# legend adı: Sicaklik 
# Çizgi kalınlığı: 2 
p.line(x, y, legend_label="Sicaklik.", line_width=2) 
# Çizimi göster 
show(p)