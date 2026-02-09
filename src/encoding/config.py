# -*- coding: utf-8 -*-
"""
Configuration management for genomic encoding.

Provides default configurations and utilities for loading/saving
training configurations.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
import json


@dataclass
class ModelConfig:
    """VAE model architecture configuration."""
    latent_dim: int = 512
    hidden_dim: List[int] = field(default_factory=lambda: [2048, 1024])
    dropout: float = 0.0
    leaky_slope: float = 0.2


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    beta: float = 1.0  # MMD weight
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class DataConfig:
    """Data paths and settings."""
    csv_path: str = ""
    label_column: str = "Majority_Subtype_mRNA"
    id_column: str = "Patient_ID"
    metadata_csv: Optional[str] = None


@dataclass
class OutputConfig:
    """Output paths and settings."""
    out_dir: str = "./output"
    mopadi_features_dir: Optional[str] = None  # Defaults to out_dir/mopadi_features
    save_checkpoints: bool = True
    checkpoint_dir: Optional[str] = None  # External checkpoint directory


@dataclass
class EncodingConfig:
    """Complete configuration for genomic encoding."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EncodingConfig":
        """Create from dictionary."""
        return cls(
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
            data=DataConfig(**d.get("data", {})),
            output=OutputConfig(**d.get("output", {})),
        )
    
    @classmethod
    def load(cls, path: str) -> "EncodingConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


def get_default_config() -> EncodingConfig:
    """Get default configuration."""
    return EncodingConfig()


# Legacy compatibility: parse old config.yaml format
def load_legacy_config(path: str) -> EncodingConfig:
    """
    Load configuration from legacy config.yaml format.
    
    Maps old format to new EncodingConfig structure.
    """
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    
    config = EncodingConfig()
    
    # Map data section
    if "data" in d:
        config.data.csv_path = d["data"].get("csv_path", "")
        config.data.id_column = d["data"].get("id_column", "Patient_ID")
    
    # Map model section
    if "model" in d:
        config.model.latent_dim = d["model"].get("latent_dim", 512)
        config.model.hidden_dim = d["model"].get("hidden_dim", [2048, 1024])
    
    # Map training section
    if "training" in d:
        config.training.epochs = d["training"].get("num_epochs", 100)
        config.training.batch_size = d["training"].get("batch_size", 64)
        config.training.learning_rate = d["training"].get("learning_rate", 1e-3)
    
    # Map output section
    if "output" in d:
        out_path = d["output"].get("encoded_data_path", "./output")
        config.output.out_dir = str(Path(out_path).parent)
    
    return config
