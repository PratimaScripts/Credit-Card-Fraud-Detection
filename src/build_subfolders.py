from pathlib import Path

# define subfolders
inputs = os.path.join('..', 'data', '03_processed')
models_reports = os.path.join('..', 'data', '04_models')
model_outputs = os.path.join('..', 'data', '05_model_output')
reports = os.path.join('..', 'data', '06_reporting')

#build subfolders
for subfolder in [models_reports, model_outputs, reports]:
    Path(subfolder).mkdir(parents=True, exist_ok=True)