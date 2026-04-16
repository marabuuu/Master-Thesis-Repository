from __future__ import annotations

from torch.utils.data import Dataset

from .tokenizer import GeneExpressionTokenizer

try:
    from joint_training.dataset import GenomicTileDataset
except ImportError:  # pragma: no cover
    from src.joint_training.dataset import GenomicTileDataset  # type: ignore[import-not-found]


class GeneTokenizedGenomicTileDataset(Dataset):
    """Phase 0 wrapper that augments `GenomicTileDataset` with tokenized genomic fields."""

    def __init__(self, base_dataset: GenomicTileDataset, tokenizer: GeneExpressionTokenizer):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer

        if hasattr(base_dataset, "gene_names"):
            base_gene_names = tuple(base_dataset.gene_names)
            if base_gene_names != tokenizer.gene_names:
                raise ValueError("Tokenizer gene order must match base dataset gene order")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        tokens = self.tokenizer.tokenize(item["genomic"])
        return {
            **item,
            **tokens,
        }
