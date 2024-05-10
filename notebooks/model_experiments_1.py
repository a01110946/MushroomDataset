#!/usr/bin/env python
# coding: utf-8

# ## Entrenamiento de Modelo

# ### Relevancia de la Estrategia de MLOps

# La implementación de una estrategia de MLOps en el análisis del dataset de hongos cobra una importancia crucial, especialmente cuando abordamos la pregunta: *"¿podemos determinar si un hongo es comestible basándonos en sus características físicas?"*. Esta pregunta no solo plantea un desafío analítico y de modelado significativo, sino que también implica una profunda responsabilidad ética y de seguridad. En ámbitos donde la salud humana podría estar en juego, la precisión y fiabilidad de las predicciones de nuestro modelo no son meramente objetivos deseables, sino imperativos críticos.
# 
# En este contexto, una estrategia de MLOps es esencial, ya que permite la iteración continua y sistemática sobre el modelo, buscando optimizar su desempeño mientras se asegura la precisión y fiabilidad del sistema. La incorporación de MLOps facilita una mejora constante mediante la automatización, la integración y la entrega continuas, junto con el monitoreo y mantenimiento en producción, lo que es crucial para manejar la delicada naturaleza de la pregunta de investigación y garantizar la seguridad de las predicciones.
# 
# En consonancia con este enfoque iterativo y basado en la necesidad de equilibrar la precisión con la responsabilidad, elegimos comenzar nuestro proceso de modelado con un modelo base sencillo: la regresión logística. Esta elección se fundamenta en su interpretabilidad, simplicidad y eficacia probada como punto de partida en problemas de clasificación. A partir de este modelo base, podemos evaluar su desempeño como línea de base y, apoyados por la infraestructura y prácticas que MLOps facilita, proceder a experimentar y mejorar iterativamente. Esto nos permite explorar modelos más complejos y ajustar parámetros con el objetivo final de optimizar la precisión y la seguridad de nuestras predicciones, garantizando así que nuestro sistema de predicción evolucione de manera responsable y efectiva para proteger la salud y el bienestar de las personas.

# ### Importación de Librerías

# In[171]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from scipy.stats import randint
from scipy.stats import uniform

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer

import mlflow
import mlflow.sklearn

from rich import print

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="distutils")


# ### Carga del Dataset

# In[172]:


parent_directory = os.path.dirname(os.getcwd())
path_primary_data = os.path.join(parent_directory, "data", "secondary_data.csv")
df = pd.read_csv(path_primary_data, sep=";")
df.info()


# ### Validación de Datos

# In[173]:


# Verificar si hay filas duplicadas
duplicates = df.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")


# In[174]:


# Verificar si hay valores faltantes en cada columna
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)


# In[175]:


def detect_outliers_iqr(dataframe):
    """
    Detecta outliers en todas las columnas numéricas de un DataFrame usando el método del rango intercuartílico (IQR).
    
    Parámetros:
    - dataframe: DataFrame de pandas que contiene las variables numéricas.
    
    Retorna:
    - Un DataFrame que contiene solo las filas que son consideradas outliers en alguna de las columnas numéricas.
    """
    outliers_df = pd.DataFrame(columns=dataframe.columns)
    
    # Selecciona solo las columnas numéricas; en caso de que se haya ingresado un DataFrame con columnas categóricas
    numeric_cols = dataframe.select_dtypes(include=['int64', 'float64'])
    
    for column in numeric_cols:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filtra los outliers
        filter_outliers = (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
        outliers_in_column = dataframe[filter_outliers]
        
        # Agrega los outliers al DataFrame de outliers
        outliers_df = pd.concat([outliers_df, outliers_in_column], axis=0).drop_duplicates().reset_index(drop=True)
    
    return outliers_df


# In[176]:


# %%capture --no-display
# Detectando outliers en el DataFrame de variables numéricas
outliers_df = detect_outliers_iqr(df)


# ### Limpieza de Datos

# - Tratar los valores faltantes (imputación, eliminación de filas/columnas, etc.).
# - Identificar y manejar valores atípicos (outliers).
# - Manejar datos duplicados y/o inconsistentes.

# #### Eliminación de outliers

# In[177]:


# Eliminar filas con outliers del conjunto de datos original
df_no_outliers = df.drop(outliers_df.index)
df_no_outliers.info()


# #### Eliminación de Variables No Deseadas

# In[178]:


# Eliminar columnas de acuerdo con observaciones en el análisis exploratorio
features_to_drop = ['cap-shape', 'does-bruise-or-bleed', 'gill-spacing', 'gill-color', 'stem-height', 'stem-color', 'ring-type', 'habitat', 'season']
df_clean = df_no_outliers.drop(columns=features_to_drop)
df_clean.head(10)


# ### División de Datos

# #### Conjunto de Entrenamiento y Prueba

# In[179]:


# División de datos en variables independientes (X) y dependientes (y)
X = df_clean.drop('class', axis=1)  # Características o variables independientes
y = df_clean['class']  # Variable objetivo o dependiente

# División estratificada en conjunto de entrenamiento (70%), validación (15%) y prueba (15%)
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, stratify=y_val_test, random_state=42)

# Imprimir información sobre la división de datos
print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de validación:", X_val.shape)
print("Tamaño del conjunto de prueba:", X_test.shape)


# #### Conjunto de Entrenamiento: Variables Numéricas y Categóricas

# In[180]:


# Identificar características numéricas y categóricas en el conjunto de entrenamiento
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns


# ### Transformación de Datos

# In[181]:


# Definir transformadores para variables numéricas y categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Imputación utilizando la mediana, debido a la distribución sesgada de los datos
    ('standard_scaler', StandardScaler()),  # Estandarización
    ('min_max_scaler', MinMaxScaler())  # Normalización
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputación utilizando la moda
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Codificación utilizando OneHotEncoder, solo tenemos variables categóricas nominales
    # ('selector', Add a selector estimator)
    # ('dim_reducer', Add a dimensionality reduction estimator)
])


# In[182]:


# Crear el preprocesador con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ])


# In[183]:


# Especificar los modelos a entrenar y evaluar
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MLP Classifier': MLPClassifier()
}

# Crear el pipeline final con el preprocesador y el modelo de regresión logística
pipelines = {}
for model_name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipelines[model_name] = pipeline


# ### Entrenamiento y Validación del Modelo

# In[184]:


# Define la cuadrícula de hiperparámetros para cada modelo (for GridSearchCV)
"""
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.1, 1, 10],                          # Controls regularization strength (smaller C -> stronger regularization)
        'classifier__max_iter': [100, 200, 500],                # Maximum number of iterations for the solver, important in large datasets to allow data to converge
        'classifier__solver': ['liblinear', 'sag', 'saga'],     # Algorithm used for optimization; 'liblinear' is often good for smaller datasets; 'sag' or 'saga' can handle larger datasets better.
        'classifier__penalty': ['l1', 'l2']                     # Type of regularization penalty (l1 for sparser models, l2 for preventing overfitting)
    },
    'Decision Tree': {
        'classifier__max_depth': [3, 5, 7],                     # Maximum depth of the tree; deeper trees can model more complex relationships but are more prone to overfitting
        'classifier__min_samples_split': [2, 5, 10],            # Minimum number of samples required to split an internal node; higher values prevent overfitting
        'classifier__min_samples_leaf': [1, 2, 5],              # Minimum samples allowed in a leaf node; higher values prevent overfitting
        'classifier__criterion': ['gini', 'entropy']            # Function to measure the quality or impurity of a split; 'gini' for Gini impurity, 'entropy' for information gain
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200],             # Number of trees in the forest; higher values reduce overfitting
        'classifier__max_depth': [3, 5, 7],                     # Maximum depth of each tree; deeper trees can model more complex relationships but are more prone to overfitting
        'classifier__min_samples_split': [2, 5, 10],            # Minimum number of samples required to split an internal node; higher values prevent overfitting
        'classifier__min_samples_leaf': [1, 2, 5],              # Minimum samples allowed in a leaf node; higher values prevent overfitting
        'classifier__max_features': ['auto', 'sqrt', 'log2']    # Number of features to consider when looking for the best split; 'auto' uses all features, 'sqrt' uses the square root of the number of features, 'log2' uses the base-2 logarithm of the number of features
    },
    'Gradient Boosting': {
        'classifier__learning_rate': [0.01, 0.1, 1],            # Step size shrinkage used to prevent overfitting; lower values are more robust but require more trees
        'classifier__n_estimators': [50, 100, 200],             # Number of boosting stages to be run; more trees reduce overfitting
        'classifier__max_depth': [3, 5, 7],                     # Maximum depth of each tree; deeper trees can model more complex relationships but are more prone to overfitting
        'classifier__subsample': [0.5, 0.8, 1.0]                # Fraction of samples used to fit each base learner; lower values prevent overfitting
    },
    'MLP Classifier': {
        'classifier__hidden_layer_sizes': [
            (50,), (100,), (50, 50), (50, 50, 50)],             # Number of neurons in each hidden layer; deeper architectures can model more complex relationships but are more prone to overfitting
        'classifier__activation': ['relu', 'tanh'],             # Activation function for the hidden layers; 'relu' is often used for deeper networks, 'tanh' can prevent vanishing gradients
        'classifier__solver': ['adam', 'lbfgs', 'sgd'],         # Algorithm used for optimization; 'adam' is often a good choice for large datasets
        'classifier__learning_rate': [
            'constant', 'adaptive', 'invscaling']               # Learning rate schedule for weight updates; 'constant' keeps the learning rate constant, 'adaptive' keeps it constant as long as training loss keeps decreasing, 'invscaling' gradually decreases the learning rate
    }
}
"""


# In[185]:


# Define la cuadrícula de hiperparámetros para cada modelo (for RandomizedSearchCV)
param_distributions = {
    'Logistic Regression': {
        'classifier__C': uniform(loc=0.01, scale=9.99), 
        'classifier__max_iter': randint(100, 501),  
        'classifier__solver': ['liblinear', 'saga'], 
        'classifier__penalty': ['l1', 'l2'] 
    },
    'Decision Tree': {
        'classifier__max_depth': randint(3, 8),  
        'classifier__min_samples_split': randint(2, 11), 
        'classifier__min_samples_leaf': randint(1, 6), 
        'classifier__criterion': ['gini', 'entropy'] 
    },
    'Random Forest': {
        'classifier__n_estimators': randint(50, 201), 
        'classifier__max_depth': randint(3, 8), 
        'classifier__min_samples_split': randint(2, 11),
        'classifier__min_samples_leaf': randint(1, 6), 
        'classifier__max_features': ['sqrt', 'log2']
    },
    'Gradient Boosting': {
        'classifier__learning_rate': uniform(loc=0.01, scale=0.99), 
        'classifier__n_estimators': randint(50, 201), 
        'classifier__max_depth': randint(3, 8),
        'classifier__subsample': uniform(loc=0.4, scale=0.6)  
    },
    'MLP Classifier': {
        'classifier__hidden_layer_sizes': [
            (50,),
            (100,),
            (50, 50),
            (50, 50, 50)            
        ], 
        'classifier__activation': ['relu', 'tanh'], 
        'classifier__solver': ['adam', 'lbfgs', 'sgd'],  
        'classifier__learning_rate': ['constant', 'adaptive', 'invscaling'] 
    }
}


# In[186]:


# Define la cuadrícula de hiperparámetros para cada modelo (for RandomizedSearchCV; short version)
"""
param_distributions = {
    'Logistic Regression': {
        'classifier__C': uniform(loc=0.01, scale=9.99),  
        'classifier__solver': ['liblinear', 'saga'], 
        'classifier__penalty': ['l2'] 
    },
    'Decision Tree': {
        'classifier__max_depth': randint(3, 8),  
        'classifier__min_samples_split': randint(2, 11), 
        'classifier__min_samples_leaf': randint(1, 6), 
        'classifier__criterion': ['gini', 'entropy'] 
    },
    'Random Forest': {
        'classifier__n_estimators': randint(50, 201), 
        'classifier__max_depth': randint(3, 8), 
        'classifier__max_features': ['sqrt', 'log2']
    },
    'Gradient Boosting': {
        'classifier__learning_rate': uniform(loc=0.01, scale=0.99),
        'classifier__subsample': uniform(loc=0.4, scale=0.6)  
    },
    'MLP Classifier': {
        'classifier__hidden_layer_sizes': [
            (50,),
            (50, 50, 50)            
        ], 
        'classifier__activation': ['relu', 'tanh'], 
        'classifier__solver': ['adam', 'lbfgs', 'sgd']
    }
}
"""


# In[187]:


# Update the param_grids dictionary with the transformed feature names (when using GridSearchCV)
"""
for model_name, grid in param_grids.items():
    updated_grid = {}
    for param, values in grid.items():
        if param.startswith('classifier__'):
            updated_grid[param] = values
    param_grids[model_name] = updated_grid
"""

# Update the param_distributions dictionary with the transformed feature names (when using RandomizerSearchCV)
for model_name, pipeline in pipelines.items():
    updated_dist = {}
    for step, params in pipeline.steps:
        if step == 'classifier':  # Detectar el paso del clasificador
            for param, values in param_distributions[model_name].items():
                updated_dist[param] = values
    param_distributions[model_name] = updated_dist


# In[188]:


# Crea un objeto de tipo StratifiedKFold para la validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Al terminar de experimentar, aumentar n_splits a 10 para una validación cruzada más robusta
cv


# In[189]:


# Define los scorers para cada métrica
scoring = {
    'precision': make_scorer(precision_score, pos_label=1),
    'recall': make_scorer(recall_score, pos_label=1),
    'f1': make_scorer(f1_score, pos_label=1),
    'auc_roc': make_scorer(roc_auc_score, response_method='predict')
}
scoring


# In[190]:


# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)


# In[191]:


model_metrics = []

# Realiza la búsqueda de hiperparámetros y la evaluación para cada modelo
for model_name, pipeline in pipelines.items():
    start_time = time.time()  # Start the timer
    # grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=cv, scoring=scoring, refit='f1', n_jobs=-1, random_state=42)
    random_search = RandomizedSearchCV(pipeline, param_distributions[model_name], n_iter=100, cv=cv, scoring=scoring, refit='f1', n_jobs=-1, random_state=42, error_score='raise') 

    with mlflow.start_run(run_name=model_name, nested=True):
        # Entrena el modelo utilizando el conjunto de entrenamiento transformado
        random_search.fit(X_train, y_train_encoded)
        
        # Encuentra el mejor modelo de acuerdo con el Grid Search
        best_model = random_search.best_estimator_
        
        # Obtiene el preprocesador ajustado del mejor modelo
        best_preprocessor = best_model.named_steps['preprocessor']
        
        # Evalúa el mejor modelo en el conjunto de validación
        y_pred_val = best_model.predict(X_val)
        precision = precision_score(y_val_encoded, y_pred_val)
        recall = recall_score(y_val_encoded, y_pred_val)
        f1 = f1_score(y_val_encoded, y_pred_val)
        auc_roc = roc_auc_score(y_val_encoded, best_model.predict_proba(X_val)[:, 1])
        conf_matrix = confusion_matrix(y_val_encoded, y_pred_val)

        end_time = time.time()  # End the timer
        training_time = end_time - start_time  # Calculate the training time

        # Append model metrics to the list
        model_metrics.append((model_name, precision, recall, f1, auc_roc, training_time))
                
        # Registra el modelo, los parámetros y las métricas en MLflow
        mlflow.log_param("model", model_name)
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc_roc)
        mlflow.log_metric("training_time", training_time)
        mlflow.sklearn.log_model(best_model, "model")

        # Extrae nombres de features y crea el diccionario de mapeo
        feature_names = best_preprocessor.get_feature_names_out()
                
        # Imprime las métricas de evaluación y la matriz de confusión
        print(f"Model: {model_name}")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        print(f"AUC-ROC: {auc_roc}")
        print(f"Training time: {training_time:.2f} seconds")
        print("Confusion Matrix:")
        print(conf_matrix)
        print()


# ### Selección del Mejor Modelo

# In[192]:


# Comparar el rendimiento de diferentes modelos basado en las métricas de evaluación
# model_metrics = []
"""
for model_name, pipeline in pipelines.items():
    with mlflow.start_run(run_name=model_name):
        # Obtener las métricas registradas en MLflow
        run = mlflow.active_run()
        metrics = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.metrics
        model_metrics.append((model_name, metrics['f1_score'], metrics['auc_roc']))
"""

# Ordenar los modelos según el F1-score y AUC-ROC
model_metrics.sort(key=lambda x: (x[3], x[4]), reverse=True)

# Seleccionar el mejor modelo para pruebas adicionales y despliegue
best_model_name = model_metrics[0][0]
best_model = pipelines[best_model_name]

print(f"Best Model: {best_model_name}")


# ### Evaluación del Mejor Modelo en el Conjunto de Prueba

# In[193]:


# Fit the best_model pipeline on the training data
best_model.fit(X_train, y_train_encoded)

# Realizar predicciones utilizando el conjunto de prueba
y_pred = best_model.predict(X_test)


# In[194]:


print(y_pred)
y_pred.shape


# In[195]:


# Encode y_test using the same LabelEncoder
y_test_encoded = label_encoder.transform(y_test)

# Evaluar el rendimiento del mejor modelo en el conjunto de prueba
accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred)
recall = recall_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred)
auc_roc = roc_auc_score(y_test_encoded, best_model.predict_proba(X_test)[:, 1])
conf_matrix = confusion_matrix(y_test_encoded, y_pred)
class_report = classification_report(y_test_encoded, y_pred)

# Imprimir los resultados
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC-ROC:", auc_roc)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)


# In[196]:


# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[197]:


# Registrar las métricas y artefactos en MLflow
"""
with mlflow.start_run(run_name="Best Model Test Evaluation"):
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_text(class_report, "classification_report.txt")
"""


# # PENDIENTE: Configurar correctamente MLflow

# # PENDIENTE: Convertir los notebooks a scripts de Python

# ### Interpretación del Modelo y Análisis de Importancia de Características

# In[198]:


# Realizar análisis de importancia de características
# ...

# Visualizar la importancia de las características
# ...


# ### Serialización y Guardado del Modelo

# In[199]:


# Serializar y guardar el mejor modelo
# ...


# ### Monitoreo y Seguimiento del Rendimiento del Modelo

# In[200]:


# Configurar mecanismos de logging y monitoreo
# ...

# Definir métricas y umbrales para detección de anomalías
# ...


# ### Mantenimiento y Reentrenamiento del Modelo

# In[201]:


# Establecer un proceso para actualizar el modelo con nuevos datos
# ...

# Definir la frecuencia y criterios para el reentrenamiento del modelo
# ...


# 
