import os
import gzip
import pickle
import io
import pandas as pd
from typing import Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_model(filename: str):
	file_path = os.path.join(ROOT_DIR, "models", filename)
	with gzip.open(file_path, "rb") as f:
		return pickle.load(f)

def load_dataframe_from_bytes(file_content: bytes, filename: Optional[str]) -> pd.DataFrame:
	name = (filename or "").lower()
	buffer = io.BytesIO(file_content)
	if name.endswith(".csv"):
		return pd.read_csv(buffer)
	elif name.endswith(".xlsx") or name.endswith(".xls"):
		return pd.read_excel(buffer)
	elif name.endswith(".parquet"):
		return pd.read_parquet(buffer)
	else:
		raise ValueError("Unsupported file format. Please upload CSV, Excel, or Parquet files.")

def main(file_content: Optional[bytes] = None, filename: Optional[str] = None):
	model = load_model("multisim_xgb.pkl.gz")
	if file_content:
		X = load_dataframe_from_bytes(file_content, filename)
	else:
		X = pd.read_parquet(os.path.join(ROOT_DIR, "data", "external", "X_test.parquet"))
	preds = model.predict(X)
	try:
		return preds.tolist()
	except Exception:
		return [p for p in preds]

if __name__ == "__main__":
	preds = main()
	print(f"Generated {len(preds)} predictions")
