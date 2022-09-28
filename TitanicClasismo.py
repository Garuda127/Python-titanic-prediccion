import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

train_df = pd.read_csv('E:/train.csv')
test_df = pd.read_csv('E:/test.csv')
train_df.head()
train_df.info()

train_df.columns
# Hay un total de 891 pasajeros en nuestro conjunto de datos de entrenamiento.
# Dado que la columna Sobrevivido tiene datos dicrete, la media nos da el número de personas que sobrevivieron de 891, es decir, el 38%.
# La mayoría de las personas pertenecían a Pclass = 3
# La tarifa máxima pagada por un boleto fue de 512, sin embargo, los precios de la tarifa variaron mucho como podemos ver en la desviación estándar de 49
#

train_df.describe()
train_df.describe(include='O')
# Búsqueda del percantaje de los valores faltantes en el conjunto de datos del tren
train_df.isnull().sum() / len(train_df) * 100
# contamos el numero de personas por sexo
sns.countplot('Sex', data=train_df)
train_df['Sex'].value_counts()
# Comparación de la función Sexo con Sobrevivió
sns.barplot(x='Sex', y='Survived', data=train_df)
train_df.groupby('Sex', as_index=False).Survived.mean()
plt.show()
# Comparación de la función Pclass con Survived
sns.barplot(x='Pclass', y='Survived', data=train_df)
train_df[["Pclass", "Survived"]].groupby(
    ['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Comparación de la función Embarked con Survived
sns.barplot(x='Embarked', y='Survived', data=train_df)
train_df[["Embarked", "Survived"]].groupby(
    ['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plt.show()
sns.barplot(x='Parch', y='Survived', data=train_df)
train_df[["Parch", "Survived"]].groupby(
    ['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plt.show()
sns.barplot(x='SibSp', y='Survived', data=train_df)
train_df[["SibSp", "Survived"]].groupby(
    ['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plt.show()
train_df.Age.hist(bins=10, color='teal')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
print("La edad media de los pasajeros es :", int(train_df.Age.median()))
print("La edad de desviación estándar de los pasajeros es :",
      int(train_df.Age.std()))

sns.lmplot(x='Age', y='Survived', data=train_df, palette='Set1')
sns.lmplot(x='Age', y='Survived', data=train_df, hue='Sex', palette='Set1')
plt.show()
# Comprobación de valores atípicos en los datos de edad
sns.boxplot(x='Sex', y='Age', data=train_df)
# obtener la edad media según el sexo
train_df.groupby('Sex', as_index=False)['Age'].median()
plt.show()
# trazar la columna Tarifa para ver la propagación de datos
sns.boxplot("Fare", data=train_df)
plt.show()
# Comprobación de los valores medios y medianos
print("media de la tarifa es :", train_df.Fare.mean())
print("El valor medio de la tarifa es :", train_df.Fare.median())
# comencemos por dejar caer los columns que no necesitaremos
drop_list = ['Cabin', 'Ticket', 'PassengerId']

train_df = train_df.drop(drop_list, axis=1)
test_passenger_df = pd.DataFrame(test_df.PassengerId)
test_df = test_df.drop(drop_list, axis=1)

test_passenger_df.head()

# rellenar los valores de Embarked que faltan en los conjuntos de datos de train y prueba
train_df.Embarked.fillna('S', inplace=True)
# Rellenar los valores que faltan en la columna Edad
train_df.Age.fillna(28, inplace=True)
test_df.Age.fillna(28, inplace=True)
# Rellenar los valores null Fare en el conjunto de datos de prueba
test_df.Fare.fillna(test_df.Fare.median(), inplace=True)
# combinar marcos de datos de tren y prueba para trabajar con ellos simultáneamente
Combined_data = [train_df, test_df]
# extraer los distintos títulos de la columna Nombres
for dataset in Combined_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Trazado de los diversos títulos extraídos de los nombres
sns.countplot(y='Title', data=train_df)
# Refinar la función de título fusionando algunos títulos
for dataset in Combined_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Special')

    dataset['Title'] = dataset['Title'].replace(
        {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

train_df.groupby('Title', as_index=False)['Survived'].mean(
).sort_values(by='Survived', ascending=False)
# Ahora veamos la distribución de la función de título
sns.countplot(y='Title', data=train_df)

# Asignación de los nombres de título a valores numéricos
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Special": 5}
for dataset in Combined_data:
    dataset['Title'] = dataset.Title.map(title_mapping)
    dataset['Title'] = dataset.Title.fillna(0)
# Creación de una nueva función IsAlone a partir de las columnas SibSp y Parch
for dataset in Combined_data:
    dataset["Family"] = dataset['SibSp'] + dataset['Parch']
    dataset["IsAlone"] = np.where(dataset["Family"] > 0, 0, 1)
    dataset.drop('Family', axis=1, inplace=True)
train_df.head()
# quitando las columnas Nombre, SibSP y Parch
for dataset in Combined_data:
    dataset.drop(['SibSp', 'Parch', 'Name'], axis=1, inplace=True)
# Creación de otra función si el pasajero es un niño
for dataset in Combined_data:
    dataset["IsMinor"] = np.where(dataset["Age"] < 15, 1, 0)

train_df['Old_Female'] = (train_df['Age'] > 50) & (train_df['Sex'] == 'female')
train_df['Old_Female'] = train_df['Old_Female'].astype(int)

test_df['Old_Female'] = (test_df['Age'] > 50) & (test_df['Sex'] == 'female')
test_df['Old_Female'] = test_df['Old_Female'].astype(int)

# Conversión de variables categóricas en numéricas
train_df2 = pd.get_dummies(
    train_df, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)
test_df2 = pd.get_dummies(
    test_df, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)
train_df2.head()
# creación de bandas de edad
train_df2['AgeBands'] = pd.qcut(train_df2.Age, 4, labels=False)
test_df2['AgeBands'] = pd.qcut(test_df2.Age, 4, labels=False)
# creación de bandas de Fare
train_df2['FareBand'] = pd.qcut(train_df2.Fare, 7, labels=False)
test_df2['FareBand'] = pd.qcut(test_df2.Fare, 7, labels=False)
# Eliminación de las columnas Edad y Tarifa
train_df2.drop(['Age', 'Fare'], axis=1, inplace=True)
test_df2.drop(['Age', 'Fare'], axis=1, inplace=True)
train_df2.head()
# sns.barplot('AgeBands','Survived',data=train_df2)

# * Machine Learning
# importar las bibliotecas de aprendizaje automático necesarias
# Dividir los datos de entrenamiento en X: características e y: objetivo
X = train_df2.drop("Survived", axis=1)
y = train_df2["Survived"]

# dividir nuestros datos de entrenamiento de nuevo en los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# Regresión logística
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
acc_logreg
print("Regresion logistica",acc_logreg)
# Vamos a realizar una validación cruzada K-fold para la regresión logística
cv_scores = cross_val_score(logreg, X, y, cv=5)

np.mean(cv_scores)*100
# Clasificador de árbol de decisión

decisiontree = DecisionTreeClassifier()
dep = np.arange(1, 10)
param_grid = {'max_depth': dep}

clf_cv = GridSearchCV(decisiontree, param_grid=param_grid, cv=5)

clf_cv.fit(X, y)
clf_cv.best_params_, clf_cv.best_score_*100
print("arbol de decision")
print('Mejor Valor de max_depth:', clf_cv.best_params_)
print('Mejor puntuación:', clf_cv.best_score_*100)
# Aleatoriedad del clásificador de bosque

random_forest = RandomForestClassifier()
ne = np.arange(1, 20)
param_grid = {'n_estimators': ne}

rf_cv = GridSearchCV(random_forest, param_grid=param_grid, cv=5)

rf_cv.fit(X, y)
print("Random Forest")
print('Mejor Valor de n_estimators:', rf_cv.best_params_)
print('Mejor puntuación:', rf_cv.best_score_*100)


gbk = GradientBoostingClassifier()
ne = np.arange(1, 20)
dep = np.arange(1, 10)
param_grid = {'n_estimators': ne, 'max_depth': dep}

gbk_cv = GridSearchCV(gbk, param_grid=param_grid, cv=5)

gbk_cv.fit(X, y)
print('Mejor valor de los parámetros:', gbk_cv.best_params_)
print('Mejor puntuación:', gbk_cv.best_score_*100)
y_final = clf_cv.predict(test_df2)

submission = pd.DataFrame({
    "PassengerId": test_passenger_df["PassengerId"],
    "Survived": y_final
})
submission.head()
submission.to_csv('titanic.csv', index=False)


# draw a bar plot of survival by sex
#sns.barplot(x="Sex", y="Survived", data=train_df)

# print percentages of females vs. males that survive
# print("Percentage of females who survived:",
#     train_df["Survived"][train_df["Sex"] == 'female'].value_counts(normalize=True)[1]*100)

# print("Percentage of males who survived:",
#     train_df["Survived"][train_df["Sex"] == 'male'].value_counts(normalize=True)[1]*100)

temp = input()
sns.set_theme()
plt.show()
