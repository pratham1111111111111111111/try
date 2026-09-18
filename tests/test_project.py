from pathlib import Path
import py_compile


ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist():
    required_files = [
        "src/train.py",
        "src/multi_model_experiment.py",
        "api/api.py",
        "Dockerfile",
        "requirements.txt",
    ]

    for file in required_files:
        assert (ROOT / file).exists(), f"Missing file: {file}"


def test_training_script_syntax():
    py_compile.compile(
        str(ROOT / "src/train.py"),
        doraise=True
    )


def test_mlflow_experiment_script_syntax():
    py_compile.compile(
        str(ROOT / "src/multi_model_experiment.py"),
        doraise=True
    )


def test_api_script_syntax():
    py_compile.compile(
        str(ROOT / "api/api.py"),
        doraise=True
    )


def test_dockerfile_exists():
    dockerfile = ROOT / "Dockerfile"
    assert dockerfile.exists()
    assert dockerfile.stat().st_size > 0