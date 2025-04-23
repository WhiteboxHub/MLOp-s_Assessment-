import os
from pathlib import Path

# Define the folder structure and files within each
structure = {
    "forecast-model-api": {
        "api": [
            "main.py",
            "model.py",
            "schema.py",
            "utils.py"
        ],
        "model_training": [
            "train_model.ipynb",
            "train.py",
            "model.pkl"
        ],
        "data": [],  # Folder only, usually .gitignored
        ".": [  # Files directly under forecast-model-api
            "Dockerfile",
            "requirements.txt",
            ".gitignore",
            "README.md"
        ]
    }
}

def create_structure(root, tree):
    for folder, contents in tree.items():
        root_path = Path(root) / folder
        root_path.mkdir(parents=True, exist_ok=True)

        for subfolder, files in contents.items() if isinstance(contents, dict) else [(".", contents)]:
            current_dir = root_path / subfolder if subfolder != "." else root_path
            current_dir.mkdir(parents=True, exist_ok=True)

            for file in files:
                file_path = current_dir / file
                file_path.touch()  # Create an empty file

# Create the structure
create_structure(".", structure)
print("✅ Project folder structure created successfully.")
