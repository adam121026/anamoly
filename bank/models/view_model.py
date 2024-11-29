# %%
import joblib
import os
import glob

def load_model(model_path):
    """Load a model from the specified file path."""
    return joblib.load(model_path)

def run_model(model, data):
    """Run the model on the provided data."""
    return model.predict(data)

if __name__ == "__main__":
    data = [...]  # Replace with your input data

    report = {}

    models_dirs = glob.glob('*_models')

    for dir_path in models_dirs:
        model_files = glob.glob(os.path.join(dir_path, '*'))
        for model_file in model_files:
            model = load_model(model_file)
            predictions = run_model(model, data)
            report[model_file] = predictions

    print("Report:")
    for model, preds in report.items():
        print(f"{model}: {preds}")

# %%
