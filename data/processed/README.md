# Processed Data

Run `../venv/bin/python src/preprocessing.py` from the project root to generate the
feature-engineered train/dev/test CSV files used for inspection and grading. The model
training scripts also build their task-specific feature matrices directly from
`data/raw/`.
