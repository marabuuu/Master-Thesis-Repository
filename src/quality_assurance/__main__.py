#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry point for running quality_assurance as a module.

Usage:
    python -m quality_assurance --config src/config.yaml
    python -m src.quality_assurance --config src/config.yaml
"""

from .run_evaluation import main

if __name__ == "__main__":
    main()
