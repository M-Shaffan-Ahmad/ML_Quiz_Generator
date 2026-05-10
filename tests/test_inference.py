from pathlib import Path


def test_project_artifacts_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "data" / "raw" / "test.csv").exists()
    assert (root / "data" / "processed" / "preprocessing_summary.csv").exists()
    assert (root / "models" / "model_a" / "traditional" / "lr_model_final.joblib").exists()
    assert (root / "models" / "model_a" / "neural").exists()
    assert (root / "models" / "model_b" / "traditional" / "model_b_artifacts.joblib").exists()
    assert (root / "models" / "model_b" / "neural").exists()
    assert (root / "src" / "preprocessing.py").exists()
    assert (root / "src" / "model_a_train.py").exists()
    assert (root / "src" / "model_b_train.py").exists()
    assert (root / "src" / "inference.py").exists()
    assert (root / "src" / "evaluate.py").exists()
    assert (root / "ui" / "app.py").exists()
    assert (root / "ui" / "components").exists()
    assert (root / "notebooks" / "EDA.ipynb").exists()
    assert (root / "notebooks" / "experiments.ipynb").exists()
    assert (root / "report" / "final_report.pdf").exists()


if __name__ == "__main__":
    test_project_artifacts_exist()
    print("project artifact smoke test passed")
