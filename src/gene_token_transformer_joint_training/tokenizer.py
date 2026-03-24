from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GeneExpressionTokenizer:
    """Phase 0 tokenizer scaffold for deterministic gene-token ordering."""

    gene_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.gene_names:
            raise ValueError("gene_names must not be empty")

    @property
    def vocab_size(self) -> int:
        return len(self.gene_names)

    def tokenize(self, genomic: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert a dense genomic vector into token ids + values.

        Expected input shape: [L] where L == len(gene_names).
        """
        if genomic.ndim != 1:
            raise ValueError("genomic tensor must be rank-1 for tokenization")
        if genomic.shape[0] != self.vocab_size:
            raise ValueError(
                f"genomic length {genomic.shape[0]} does not match tokenizer vocab {self.vocab_size}"
            )

        gene_ids = torch.arange(self.vocab_size, dtype=torch.long)
        gene_values = genomic.to(torch.float32)
        attention_mask = torch.ones(self.vocab_size, dtype=torch.bool)
        return {
            "gene_ids": gene_ids,
            "gene_values": gene_values,
            "attention_mask": attention_mask,
        }
