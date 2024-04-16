def get_feature_names_out_with_mapping(preprocessor):
  """
  Extrae los nombres de las features transformadas y crea un diccionario con la 
  correspondencia entre nombres originales y transformados.

  Parámetros:
    preprocessor: Objeto ColumnTransformer ya ajustado.

  Retorno:
    - feature_names: Lista con nombres de features transformadas.
    - feature_mapping: Diccionario con la correspondencia entre nombres originales 
      y transformados.
  """
  feature_names = preprocessor.get_feature_names_out()
  feature_mapping = {}

  for i, name in enumerate(feature_names):
    if name.startswith('numeric__'):
      original_name = name.split('__')[1]
      feature_mapping[i] = original_name
    elif name.startswith('categorical__'):
      original_name = name.split('__')[1].split('_')[0]
      feature_mapping[i] = original_name

  return feature_names, feature_mapping
