from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd

# URL de los datos:

url = "https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv"

# Cargar los datos desde la URL:

datos_csv = pd.read_csv(url)

# Guardar los datos en un archivo CSV en la carpeta ./data/raw:

datos_csv.to_csv("../data/raw/datos_brutos.csv", index = False)


# obtener las dimensiones:

datos_csv.shape

# obtener información sobre tipos de datos y valores nulos:

datos_csv.info()

# Eliminar duplicados:

#1 contar las filas duplicadas en el data frame:

num_duplicados = datos_csv.drop_duplicates().T.drop_duplicates().T
print("Número de valores duplicados antes de eliminarlos:", num_duplicados)

# Eliminar información irrelevante: id, name, host_name , reviews_per_month:

datos_csv.drop(["id", "name", "host_name", "reviews_per_month"], axis = 1, inplace = True)
datos_csv.head()

# crear un date frame nuevo con los datos limpios:

#datos_csv_limpios.to = datos_csv[]

# Paso 3: Análisis de variables univariantes:

# Crear una figura y ejes para los subgráficos:

import matplotlib.pyplot as plt
import seaborn as sns

fig, axis = plt.subplots(2, 3, figsize=(15, 15))

# Crear histogramas para diferentes variables:



sns.histplot(ax=axis[0,0], data=datos_csv, x="room_type")
sns.histplot(ax=axis[0,1], data=datos_csv, x="longitude").set_xticks([])
sns.histplot(ax=axis[0,2], data=datos_csv, x="host_id").set_xticks([])
sns.histplot(ax=axis[1,0], data=datos_csv, x="number_of_reviews")
sns.histplot(ax=axis[1,1], data=datos_csv, x="price")
fig.delaxes(axis[1, 2])

# Ajustar el diseño de los subgráficos:

plt.tight_layout()

# Mostrar el gráfico:

plt.show()

# Análisis de Variables numéricas:

# Crear una figura y ejes para los subgráficos: 

fig, axis = plt.subplots(4, 2, figsize=(15, 15), gridspec_kw={"height_ratios": [6, 1, 6, 1]})

# Crear histogramas y boxplots para diferentes variables numéricas:

sns.histplot(ax=axis[0, 0], data=datos_csv, x="availability_365")
sns.boxplot(ax=axis[1, 0], data=datos_csv, x="availability_365")

sns.histplot(ax=axis[0, 1], data=datos_csv, x="minimum_nights").set_xlim(0, 200)
sns.boxplot(ax=axis[1, 1], data=datos_csv, x="minimum_nights")

sns.histplot(ax=axis[2, 0], data=datos_csv, x="number_of_reviews")
sns.boxplot(ax=axis[3, 0], data=datos_csv, x="number_of_reviews")

sns.histplot(ax=axis[2, 1], data=datos_csv, x="price")
sns.boxplot(ax=axis[3, 1], data=datos_csv, x="price")

# Ajustar el diseño de los subgráficos:

plt.tight_layout()

# Mostrar el gráfico:

plt.show()

# Paso 5: Análisis de Variables multivariadas:

# Análisis numérico-numérico:

# Crear una figura y ejes para los subgráficos:

fig, axis = plt.subplots(4, 2, figsize=(15, 15))

# Crear gráficos de dispersión y mapas de calor para diferentes variables numéricas:

sns.regplot(ax=axis[0, 0], data=datos_csv, x="minimum_nights", y="number_of_reviews")
sns.heatmap(datos_csv[["number_of_reviews", "minimum_nights"]].corr(), annot=True, fmt=".2f", ax=axis[1, 0], cbar=False)

sns.regplot(ax=axis[0, 1], data=datos_csv, x="availability_365", y="price").set(ylabel=None)
sns.heatmap(datos_csv[["price", "availability_365"]].corr(), annot=True, fmt=".2f", ax=axis[1, 1])

sns.regplot(ax=axis[2, 0], data=datos_csv, x="minimum_nights", y="price").set(ylabel=None)
sns.heatmap(datos_csv[["price", "minimum_nights"]].corr(), annot=True, fmt=".2f", ax=axis[3, 0]).set(ylabel=None)

# Eliminar subgráficos no utilizados:

fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])

# Ajustar el diseño de los subgráficos:

plt.tight_layout()

# Mostrar el gráfico.

plt.show()

# Paso 6: Análisis categórico-categórico:

# Crear una figura y ejes para el subgráfico:

fig, axis = plt.subplots(figsize=(10, 10))

# Crear un diagrama de conteo con agrupación por "room_type" para el precio ("price"):

sns.countplot(data=datos_csv, x="neighbourhood_group", hue="room_type")

# Mostrar el gráfico:

plt.show()

#Paso 7: Análisis numérico-categórico completo:

# Factorizar los datos de tipo de habitación y vecindario:

datos_csv["room_type"] = pd.factorize(datos_csv["room_type"])[0]
datos_csv["neighbourhood_group"] = pd.factorize(datos_csv["neighbourhood_group"])[0]
datos_csv["neighbourhood"] = pd.factorize(datos_csv["neighbourhood"])[0]

# Crear un mapa de calor para el análisis numérico-categórico completo:

fig, axes = plt.subplots(figsize=(15, 15))
sns.heatmap(datos_csv[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",    
                        "number_of_reviews", "calculated_host_listings_count", "availability_365"]].corr(), annot=True, fmt=".2f")

plt.tight_layout()

# Mostrar el gráfico:

plt.show()

# Paso 8: Dibujar el Pairplot:

sns.pairplot(data=datos_csv)

# Ingenieria de características:

# Análisis descriptivo:

descriptivo = datos_csv.describe()

print(descriptivo)

# Valores Atípicos:

# Crear una figura y ejes para los subgráficos:

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Trazar diagramas de caja para diferentes variables:

sns.boxplot(ax=axes[0, 0], data=datos_csv, y="neighbourhood_group")
sns.boxplot(ax=axes[0, 1], data=datos_csv, y="price")
sns.boxplot(ax=axes[0, 2], data=datos_csv, y="minimum_nights")
sns.boxplot(ax=axes[1, 0], data=datos_csv, y="number_of_reviews")
sns.boxplot(ax=axes[1, 1], data=datos_csv, y="calculated_host_listings_count")
sns.boxplot(ax=axes[1, 2], data=datos_csv, y="availability_365")
sns.boxplot(ax=axes[2, 0], data=datos_csv, y="room_type")


plt.tight_layout()

# Mostrar el gráfico:

plt.show()

# Paso 9:

# valores atípicos para la columna "price":

# Obtener estadísticas descriptivas para la columna "price":

stats_price = datos_csv["price"].describe()

# Mostrar las estadísticas descriptivas para "price":

print("Estadísticas descriptivas para 'price':")
print(stats_price)

#Paso 10:

# Calcular rango intercuantílico y limites superiores e inferiores para la detección de valores atípicos en "price":

# Calcular el rango intercuartílico (IQR)
price_iqr = stats_price["75%"] - stats_price["25%"]

# Calcular los límites para identificar valores atípicos
upper_limit = stats_price["75%"] + 1.5 * price_iqr
lower_limit = stats_price["25%"] - 1.5 * price_iqr

# Mostrar los resultados
print(f"Los límites superior e inferior para la búsqueda de outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, con un rango intercuartílico de {round(price_iqr, 2)}")

# Limpieza de valores atípicos en "price" eliminar filas con valor mayor que 0:

datos_csv = datos_csv[datos_csv["price"] > 0]

# Contar el número de filas donde el precio es igual a 0 y 1 respectivamente:

conteo_0 = datos_csv[datos_csv["price"] == 0].shape[0]
conteo_1 = datos_csv[datos_csv["price"] == 1].shape[0]

# Imprimir el conteo de valores iguales a 0 y 1:

print("Cantidad de valores 0:", conteo_0)
print("Cantidad de valores 1:", conteo_1)

# Valor faltante:

# Contar NaN en el DataFrame "datos_csv"
conteo_nan = datos_csv.isnull().sum().sort_values(ascending=False)

# Mostrar el conteo de NaN en cada columna
print(conteo_nan)

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# lista de variables a normalizar:

num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", 
                 "availability_365", "neighbourhood_group", "room_type"]

# instancia del escalador Min-Max:

scaler = MinMaxScaler()

#Normalizar las variables de las caracteristicas seleccionadas:

scal_features = scaler.fit_transform(datos_csv[num_variables])


#Crear un nuevo DataFrame normalizado:

df_normalizado = pd.DataFrame(scal_features, index = datos_csv.index, columns = num_variables)

#Añadir la columna "price" del dataframe original al nuevo normalizado:

df_normalizado["price"] = datos_csv["price"]

# Eliminar la columna "last_review":

datos_error_dates = datos_csv.drop("last_review", axis=1)

#Mostrar las primeras lineas del dataframe normalizado:

df_normalizado.head()


# Selección de funciones:

# Eliminar la columna de fechas
#datos_csv_clean = df_normalizado.drop("last_review", axis=1)

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split



# Separar las características (X) y la variable objetivo (y)
X = df_normalizado.drop("price", axis=1)
y = df_normalizado["price"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de selección de características utilizando chi-cuadrado
selection_model = SelectKBest(chi2, k=4)

# Ajustar el modelo a los datos de entrenamiento
selection_model.fit(X_train, y_train)

# Obtener los índices de las características seleccionadas
ix = selection_model.get_support()

# Transformar los conjuntos de entrenamiento y prueba utilizando solo las características seleccionadas
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns=X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns=X_test.columns.values[ix])

# Añadir la columna de precios al conjunto de entrenamiento seleccionado
X_train_sel["price"] = list(y_train)

# Añadir la columna de precios al conjunto de prueba seleccionado
X_test_sel["price"] = list(y_test)

# Guardar el conjunto de entrenamiento seleccionado en un archivo CSV
X_train_sel.to_csv("../data/processed/clean_train.csv", index=False)

# Guardar el conjunto de prueba seleccionado en un archivo CSV
X_test_sel.to_csv("../data/processed/clean_test.csv", index=False)