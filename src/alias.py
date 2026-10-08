import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri('http://127.0.0.1:5000')
c = MlflowClient()

c.set_registered_model_alias('fraude', 'production', 1)   # meilleur F1
c.set_registered_model_alias('fraude', 'staging', 2)      # candidat

for alias in ('production', 'staging'):
    v = c.get_model_version_by_alias('fraude', alias)
    print('alias', alias, '-> version', v.version)