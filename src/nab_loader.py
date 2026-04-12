from pathlib import Path
import json
import pandas as pd


class NABLoader:
    def __init__(self, nab_root):
        self.nab_root = Path(nab_root)
        self.labels_path = self.nab_root / "combined_labels.json"

        if not self.nab_root.exists():
            raise FileNotFoundError(f"NAB folder not found: {self.nab_root}")

        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

    def list_series(self, prefix=None):
        keys = sorted(self.labels.keys())
        if prefix is not None:
            keys = [k for k in keys if k.startswith(prefix)]
        return keys

    def load_series(self, relative_path):
        candidate_paths = [
            self.nab_root / relative_path,
            self.nab_root / Path(relative_path).parts[0] / relative_path,
        ]

        csv_path = None
        for path in candidate_paths:
            if path.exists():
                csv_path = path
                break

        if csv_path is None:
            raise FileNotFoundError(
                "Series file not found. Tried:\n" + "\n".join(str(p) for p in candidate_paths)
            )

        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        label_times = pd.to_datetime(self.labels.get(relative_path, []))

        return df, label_times