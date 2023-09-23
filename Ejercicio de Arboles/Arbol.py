import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

datos = pd.read_csv(r"C:\Users\mcpa-\Documents\UNAD_2023\SEGUNDO SEMESTRE\ANALISIS DE DATOS\Etapa2\Ejercicio de Arboles\wine.data")
datos.head()

datos.shape

datos.describe()

plt.hist(datos,1)

Arbol = DecisionTreeClassifier()

Color = ["14.23", "1.71", "2.43", "15.6","127", "2.8", "3.06", ".28","2.29","5.64","1.04","3.92","1065"]
col = ['1']

predictores = datos[Color]
target = datos[col]

X_train, X_test, y_train, y_test = train_test_split(predictores, target, test_size=0.2, random_state=13)

tree = DecisionTreeClassifier()

arbol = tree.fit(X_train, y_train)
plot_tree(arbol)

predicciones = arbol.predict(X_test)

pd.crosstab(np.array([y[0] for y in y_test.values.tolist()]), predicciones, rownames=['Actual'], colnames=['Predicciones'])

accuracy = accuracy_score(y_test,predicciones)
accuracy