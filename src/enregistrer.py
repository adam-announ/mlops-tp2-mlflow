import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri('http://127.0.0.1:5000')
client = MlflowClient()

exp = client.get_experiment_by_name('tp2-registre')
runs = client.search_runs([exp.experiment_id],
                          order_by=['metrics.f1 DESC'], max_results=3)

for r in runs:
    # le modele journalise est retrouve via son run_id
    logged = client.search_logged_models(
        experiment_ids=[exp.experiment_id],
        filter_string=f"source_run_id = '{r.info.run_id}'")
    uri = f"models:/{logged[0].model_id}"
    mv = mlflow.register_model(uri, 'fraude')
    print('version', mv.version, '<- run', r.info.run_name,
          '| F1 =', round(r.data.metrics['f1'], 4))