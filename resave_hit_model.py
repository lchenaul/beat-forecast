import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

candidates = [
    BASE_DIR / "models" / "hit_gb_pipeline_wade_compat.joblib",
    BASE_DIR / "hit_gb_pipeline_wade_compat.joblib",
    BASE_DIR / "hit_gb_pipeline_wade.joblib",
]

model_path = None
for p in candidates:
    if p.exists():
        model_path = p
        break

if model_path is None:
    raise FileNotFoundError("Could not find hit model file to resave.")

print(f"Loading model from: {model_path}")
model = joblib.load(model_path.as_posix())

print("Resaving model with current local environment...")
joblib.dump(model, model_path.as_posix())

print(f"Done. Resaved: {model_path}")