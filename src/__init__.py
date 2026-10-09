# src/__init__.py
"""
PUC Chatbot package initializer.

Purpose:
- Marks `src` as a Python package.
- Exposes a small, safe public surface so other code can do e.g.
    from src import helper, prompt, topics, entities
  without triggering heavy imports on package import.
- Keeps the package lightweight: do NOT import heavy modules (langchain, spacy, etc.)
  at module import time to avoid startup delays or side-effects.

Notes / Changes made:
- Added __all__ to explicitly document public modules.
- Added __version__ so the package can be versioned if needed.
- No heavy imports are performed here by design (to keep imports cheap).
"""

__all__ = ["helper", "prompt", "topics", "entities"]

# Semantic version for this package — bump when making breaking changes.
__version__ = "0.1.0"
