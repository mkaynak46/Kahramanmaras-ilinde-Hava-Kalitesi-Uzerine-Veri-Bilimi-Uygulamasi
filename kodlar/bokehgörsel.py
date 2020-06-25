# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 01:56:05 2020

@author: Musty Kaynak
"""

# Gerekli kütüphane
import pandas as pd
from bokeh.plotting import figure, output_file, show
data=pd.read_excel("KahramanmarasVeriset.xlsx")

df = pd.DataFrame(data)
print(df)
# verilerim
x = data.PM10
y0 = data.PM10Debi
y1 = data.Bagilnem
y2 = data.Havasicakligi

# html çıktının adı
output_file("log_cizgi.html")

# cizimi olustur
# tools ile hangi araclar eklenecek
# y_axis_type ile y eksen logaritmik artacak
# y_range ile eksen aralığı belirtiliyor

p = figure(
   tools="pan,box_zoom,reset,save",
   y_axis_type="log", y_range=[0.001, 1000], title="logaritmik gösterim",
   x_axis_label='x ekseni', y_axis_label='y ekseni'
)

#  çizgi ve scatter (nokta)
p.line(x, x, legend_label="PM10")
p.circle(x, x, legend_label="PM10", fill_color="white", size=8)
p.line(x, y0, legend_label="PM10Debi", line_width=3)
p.line(x, y1, legend_label="Bagilnem", line_color="red")
p.circle(x, y1, legend_label="Bagilnem", fill_color="red", line_color="red", size=6)
p.line(x, y2, legend_label="Havasicakligi", line_color="orange", line_dash="4 4")

# Çizimi göster
show(p)