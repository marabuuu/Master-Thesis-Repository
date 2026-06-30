"""
Utility functions shared across quality_assurance modules.

Consolidates common patterns like patient ID extraction.
"""

import re
from pathlib import Path
from typing import Optional


def extract_patient_id(name: str) -> str:
    """Extract TCGA patient identifier from a filename or string.
    
    Handles both patterns:
    - Long SVS-derived names: ``patient1-DX1.UUID.HASH.zip``
    - Short reconstructed names: ``patient1.zip``
    - Any string containing TCGA pattern
    
    Parameters
    ----------
    name : str
        Filename or string potentially containing TCGA identifier
        
    Returns
    -------
    str
        Extracted TCGA identifier (e.g., ``patient1``) or 'unknown'
    """
    stem = Path(name).stem.upper()
    
    # Try TCGA regex first – most reliable
    m = re.match(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)", stem)
    if m:
        return m.group(1)
    
    # Fallback: normalise separators and extract first 3 TCGA parts
    for sep in ("_", "."):
        stem = stem.replace(sep, "-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3])
    
    # Last resort: case-insensitive search anywhere in original string
    m = re.search(r"(?i)(TCGA-[A-Z0-9]+-[A-Z0-9]+)", name)
    if m:
        return m.group(1).upper()
    
    return "unknown"
