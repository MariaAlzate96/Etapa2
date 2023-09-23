#Regresion  Logistica 
import matplotlib.pyplot as plt
from sklearn import linear_model # usando sklear para saber los valores optimos
import seaborn as sns
import numpy as np
import pandas as pd


datos=pd.read_csv(r"C:\Users\mcpa-\Documents\UNAD_2023\SEGUNDO SEMESTRE\ANALISIS DE DATOS\Etapa2\Ejercicio de Regresion Logistica\framingham.csv", sep=",")

# Seleccion de Datos de Indce de masa corporal y diabetes 

datos[['BMI','TenYearCHD']].head()

datos[['BMI','TenYearCHD']].plot.scatter(x='BMI', y='TenYearCHD')

#pruebas 
w=0.09
b= -3.6

# puntos en la recta 
x= np.linspace(0,datos['BMI'].max(),100)
y= 1/(1+np.exp(-(w*x+b)))

datos.plot.scatter(x="BMI", y="TenYearCHD", label='Datos Tabla')
plt.plot(x, y, color='black', label='Línea de Logistica')
plt.ylim(0,datos['TenYearCHD'].max()*1.1)
plt.scatter(x, y, color='red')
plt.xlabel('BMI')
plt.ylabel('TenYearCHD')
plt.legend()
plt.show()


