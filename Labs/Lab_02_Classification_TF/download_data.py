"""Скачивает папку satellite из датасета andrewcropper/ilp-datasets (Hugging Face) в data/."""
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(__file__).resolve().parent / "data"
snapshot_download(
    repo_id="andrewcropper/ilp-datasets",
    repo_type="dataset",
    allow_patterns=["satellite/*", "satellite/**"],
    local_dir=target,
)
print("Готово:", target / "satellite")
