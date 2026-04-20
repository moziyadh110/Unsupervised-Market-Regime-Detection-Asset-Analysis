"""
Geopolitical Risk Analysis - Modular data science package
Analyzing impact of geopolitical risks on gold and silver prices
"""

__version__ = "1.0.0"
__author__ = "Mohamed Ziyadh Mohamed"

from . import data_loader
from . import preprocessing
from . import feature_engineering
from . import models
from . import evaluation
from . import visualization

__all__ = [
    'data_loader',
    'preprocessing',
    'feature_engineering',
    'models',
    'evaluation',
    'visualization'
]
