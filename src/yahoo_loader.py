from pathlib import Path
from typing import List, Tuple

import pandas as pd


class YahooLoader:
    """
    Minimal loader for locally downloaded Yahoo anomaly CSV files.

    Expected file format:
        timestamp,value,is_anomaly

    Example folder:
        Yahoo/
            real_1.csv
            real_2.csv
            ...
    """

    def __init__(self, yahoo_root: str | Path):
        self.yahoo_root = Path(yahoo_root)

        if not self.yahoo_root.exists():
            raise FileNotFoundError(f"Yahoo folder not found: {self.yahoo_root}")

    def list_series(self, prefix: str | None = None) -> List[str]:
        """
        Return sorted CSV filenames under the Yahoo root directory.

        Parameters
        ----------
        prefix : str | None
            Optional filename prefix filter, e.g. "real_".

        Returns
        -------
        List[str]
            Sorted list of matching CSV filenames.
        """
        files = sorted([p.name for p in self.yahoo_root.glob("*.csv")])

        if prefix is not None:
            files = [f for f in files if f.startswith(prefix)]

        return files

    def load_series(self, relative_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load one Yahoo series and return the cleaned dataframe together with
        anomaly timestamps.

        Parameters
        ----------
        relative_path : str
            CSV filename relative to the Yahoo root folder.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            - DataFrame with columns: timestamp, value, is_anomaly
            - Series of timestamps where is_anomaly == 1
        """
        csv_path = self.yahoo_root / relative_path

        if not csv_path.exists():
            raise FileNotFoundError(f"Series file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        required_cols = {"timestamp", "value", "is_anomaly"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {csv_path.name}: {sorted(missing)}"
            )

        out = df.copy()

        # Yahoo timestamps in the local files are integer steps. We convert them
        # to equally spaced datetimes so the downstream evaluation pipeline can
        # use the same timestamp-based logic as in the NAB experiments.
        out["timestamp"] = pd.to_datetime(out["timestamp"], unit="h", origin="unix")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out["is_anomaly"] = (
            pd.to_numeric(out["is_anomaly"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        label_times = out.loc[out["is_anomaly"] == 1, "timestamp"].reset_index(drop=True)

        return out[["timestamp", "value", "is_anomaly"]], label_times