from __future__ import annotations
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union, Any, Callable
from loguru import logger


def build_valid_subset(
    q_df: pd.DataFrame, 
    db_df: pd.DataFrame, 
    similarity_function: Callable[[dict, dict], bool],
    **similarity_kwargs: dict[str, Any]) -> list[int]:
        """Build valid subset of queries.
        
        We consider a query valid if it has at least one database pose within the recall threshold.

        Args:
            query_xyz: (N, 3) array of query positions
            index: FaissFlatIndex

        Returns:
            list[int]: valid query indices
        """
        valid: list[int] = []
        for i, q in tqdm(q_df.iterrows(), desc="Building valid subset for queries"):
            for j, db in db_df.iterrows():
                if similarity_function(q.to_dict(), db.to_dict(), **similarity_kwargs):
                    valid.append(i)
                    break
        return valid

class PRDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.df = pd.DataFrame()

    def __len__(self) -> int:  
        return int(len(self.df))

    def save_meta_parquet(
        self,
        meta_dir: Optional[Union[str, Path]] = None,
        meta_file: str = "meta.parquet"
        ):

        out = (Path(meta_dir) / meta_file) if meta_dir is not None else (self.dataset_root / meta_file)

        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(out, index=False)
        logger.info(f"Wrote {len(self.df):,} rows to {out}")
        return out

    def similarity_check(self, a: dict[str, Any], b: dict[str, Any], **kwargs) -> bool:
        raise NotImplementedError("Subclasses must implement `similarity_check(a, b, **kwargs)`.")

    def sample_from_position(self, pos: int) -> dict[str, Any]:
        raw = self.df.iloc[pos].to_dict()
        if not isinstance(raw, dict):
            raise TypeError("Dataset __getitem__ must return a dict.")
        return raw