"""Basic tests for the garbage classification app"""
import os


def test_required_files_exist():
    """Test that required files exist in the repository"""
    assert os.path.exists("app.py"), "Main app.py file should exist"
    assert os.path.exists("model.py"), "model.py file should exist"
    assert os.path.exists("class_indices.txt"), \
        "class_indices.txt should exist"
    assert os.path.exists("garbage_classifier.h5"), \
        "Model file should exist"


def test_class_indices_format():
    """Test that class_indices.txt has the correct format"""
    with open("class_indices.txt", "r") as f:
        lines = f.readlines()

    # Should have at least one line
    assert len(lines) > 0, "class_indices.txt should not be empty"

    # Each line should have format "class_name,index"
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            parts = line.split(",")
            assert len(parts) == 2, f"Invalid format in line: {line}"
            assert parts[1].isdigit(), f"Index should be numeric: {line}"


def test_requirements_file_exists():
    """Test that requirements.txt exists"""
    assert os.path.exists("requirements.txt"), "requirements.txt should exist"


def test_dockerfile_exists():
    """Test that Dockerfile exists for containerization"""
    assert os.path.exists("Dockerfile"), "Dockerfile should exist"
