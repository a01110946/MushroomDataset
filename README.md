
# Estructura de proyecto propuesta

---------------------------------------------------------------------------------------------------------------------------

{{ cookiecutter.repo_name }}/
├── LICENSE
├── README.md
├── Makefile                                   # Makefile with commands like `make data` or `make train`
├── configs                                     # Config files (models and training hyperparameters)
│   └── model1.yaml
│
├── data
│   ├── external                               # Data from third party sources.
│   ├── interim                                 # Intermediate data that has been transformed.
│   ├── processed                           # The final, canonical data sets for modeling.
│   └── raw                                      # The original, immutable data dump.
│
├── docs                                         # Project documentation.
│
├── models                                      # Trained and serialized models.
│   ├── baseline_model.pkl             # The baseline model, to compare against subsequent models.
│   └── model_1.pkl                         # The following trained model.
│
├── notebooks                                 # Jupyter notebooks.
│   ├── EDA.ipynb                            # Data from third party sources.
│   ├── baseline_model.ipynb          # Intermediate data that has been transformed.
│   ├── model_1.ipynb                     # The final, canonical data sets for modeling.
│   └── models_evaluation.ipynb     # The original, immutable data dump.
│
├── references                                 # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                                      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                                 # Generated graphics and figures to be used in reporting.
│
├── requirements.txt                       # The requirements file for reproducing the analysis environment.
└── src                                            # Source code for use in this project.
    ├── __init__.py                           # Makes src a Python module.
    │
    ├── data                                     # Data engineering scripts.
    │   ├── build_features.py
    │   ├── cleaning.py
    │   ├── ingestion.py
    │   ├── labeling.py
    │   ├── splitting.py
    │   └── validation.py
    │
    ├── models                                 # ML model engineering (a folder for each model).
    │   └── model1
    │       ├── dataloader.py
    │       ├── hyperparameters_tuning.py
    │       ├── model.py
    │       ├── predict.py
    │       ├── preprocessing.py
    │       └── train.py
    │
    └── visualization                         # Scripts to create exploratory and results oriented visualizations.
        ├── evaluation.py
        └── exploration.py

---------------------------------------------------------------------------------------------------------------------------

## Carpeta raíz

- README.md: Descripción del proyecto, objetivos, requisitos, instrucciones de uso.
- LICENSE: Licencia del proyecto (por ejemplo, Apache 2.0).
- .gitignore: Archivo para ignorar archivos innecesarios en el control de versiones.
- Makefile:                     # Makefile with commands like `make data` or `make train`
- requirements.txt: Lista de dependencias del proyecto.

## Carpeta `data/`

Contiene el dataset original y el dataset preprocesado.

- secondary_mushroom_dataset.csv: Dataset original.
- dataset_procesado.csv: Dataset preprocesado para el modelado.
- otros_datasets.csv: Otros datasets relevantes (opcional).

## Carpeta `notebooks/`

Contiene los Jupyter notebooks para el análisis, modelado y evaluación.

- EDA.ipynb: Análisis Exploratorio de Datos.
- Modelo_base.ipynb: Implementación del modelo base.
- Otros_modelos.ipynb: Implementación de otros modelos (opcional).
- Evaluacion_modelos.ipynb: Evaluación y comparación de modelos.

## Carpeta `src/`

Contiene el código Python para el pipeline.

- modulo_funciones.py: Módulo con funciones comunes para el proyecto.
- pipeline.py: Script que ejecuta el pipeline completo.

## Carpeta `docs/`

Contiene el informe final y la presentación del proyecto.

- Informe_final.pdf: Informe final del proyecto.
- Presentacion.pptx: Presentación del proyecto (opcional).

# Carpeta `models/`

Contiene los modelos entrenados.

- Modelo_base.pkl: Modelo base entrenado.
- Otros_modelos.pkl: Otros modelos entrenados (opcional).

## Carpeta env/

Contiene el entorno virtual con las dependencias del proyecto.

- Entorno virtual con las dependencias del proyecto (opcional).

## Running the Pipeline

To run the pipeline and process your dataset, use the following command:

```bash
python src/pipeline.py --data_path path/to/your/dataset.csv
```

Replace path/to/your/dataset.csv with the actual path to your dataset file.
For example, if your dataset file is named secondary_data.csv and is located in the data directory, run:

```bash
python src/pipeline.py --data_path data/secondary_data.csv
```

The pipeline will load the dataset, preprocess the data, train the model, evaluate its performance, and generate predictions.
