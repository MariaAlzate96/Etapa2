#REGRESION LINEAL
#IMPORTACION DE LIBRERIAS 
import matplotlib.pyplot as plt
from sklearn import linear_model # usando sklear para saber los valores optimos
import seaborn as sns
import numpy as np
import pandas as pd

#CARGAR DATA O INFORMACION

datos=pd.read_csv(r"C:\Users\mcpa-\Documents\UNAD_2023\SEGUNDO SEMESTRE\ANALISIS DE DATOS\Etapa2\Ejercicio Regresion Lineal\data.csv", sep=",")

#ORGANIZACIONY EXPLORACION DE LOS DATOS DE LA DATA 
datos.columns
datos.metro
datos.precio


# VISUALIAZCION DE LA GRAFICA 
datos

#Realizo la grafica de dispersión
datos.plot.scatter(x="metro", y="precio")
plt.show()


regresion = linear_model.LinearRegression()

#Agrego los datos en un array o vector
metros = datos["metro"].values.reshape((-1,1))

#Ahora si creamos el modelo
modelo = regresion.fit(metros, datos["precio"])

print("Interseccion (b)", modelo.intercept_)
#imprimos la pendiente
print("Pendiente (m)", modelo.coef_)


entrada= [[15],[25]]
predicciones = modelo.predict(entrada)
print(predicciones)

datos.plot.scatter(x="y", y="priceUSD", label='Datos originales')
plt.scatter(entrada, predicciones, color='red')
plt.plot(entrada, predicciones, color='black', label='Línea de regresión')
plt.xlabel('year')
plt.ylabel('priceUSD')
plt.legend()
plt.show()









