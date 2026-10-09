# src/entities.py
"""
Entity extraction for Premier University RAG Chatbot.

Responsibilities:
- Identify relevant entities from text (courses, events, policies, etc.).
- Return structured entities for each input text.
- Keyword-based extraction is lightweight and does not require a heavy NLP model.
"""

from typing import Optional, Dict, List
import re

# Entity categories with keywords
ENTITY_KEYWORDS: Dict[str, List[str]] = {
    # Tier 1 - Core Identity & Immediate Actions
    "UNIVERSITY": ["premier university", "pucc", "b.sc cse"],
    "COURSE": ["cse 400"],
    "EVENT": ["cse fest", "icpc"],
    "POLICY": ["admission requirements", "active membership", "code of conduct"],

    # Tier 2 - Curriculum & Wings
    "CURRICULUM": ["data structures", "algorithm design", "ai", "ml", "computer networks"],
    "WING": ["linux-based networking wing", "career guidance wing"],
    "ACTIVITY": ["bootcamps", "workshops"],

    # Tier 3 - Detailed Topics & Governance
    "DETAILED_TOPIC": [
        "convolutional neural network", "fuzzy logic", "matlab", "plc",
        "aws lambda", "mou", "sub-executive committee", "document coordinators",
        "bloom's taxonomy"
    ]
}


def extract_entities(
    text: str,
    entities_map: Optional[Dict[str, List[str]]] = None
) -> List[Dict[str, str]]:
    """
    Extract entities from text using simple keyword matching.

    Args:
        text: Input text to analyze.
        entities_map: Optional dictionary mapping entity labels to keywords.
                      Defaults to ENTITY_KEYWORDS.

    Returns:
        A list of dictionaries, each with:
        - "text": matched keyword
        - "label": entity category
    """
    if not text:
        return []

    text_lower = text.lower()
    entities: List[Dict[str, str]] = []
    entities_map = entities_map or ENTITY_KEYWORDS

    for label, keywords in entities_map.items():
        for kw in keywords:
            # match keyword exactly using word boundaries
            if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text_lower):
                entities.append({"text": kw, "label": label})

    return entities
