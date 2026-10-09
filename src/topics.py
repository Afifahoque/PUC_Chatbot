# src/topics.py
"""
Topic detection for Premier University RAG Chatbot.

Responsibilities:
- Maintain a dictionary of topics and associated keywords.
- Detect the most relevant topic for a given text using keyword matching.
- Return None if no topic matches.

Notes / changes made:
- Added docstrings for clarity.
- Used safe .lower() and regex word-boundary matching.
- The function is robust to empty or None input.
"""

from typing import Optional, Dict, List
import re

# Mapping of topic names to list of associated keywords
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    # Tier 1 - Core Identity & Immediate Actions
    "Core Identity": [
        "premier university", "pucc", "b.sc cse",
        "admission requirements", "cse fest", "icpc",
        "active membership", "code of conduct", "cse 400"
    ],

    # Tier 2 - Core Curriculum & Extracurricular Paths
    "Curriculum & Wings": [
        "data structures", "algorithm design", "ai", "ml",
        "computer networks", "linux-based networking wing",
        "career guidance wing", "peo3", "po10",
        "bootcamps", "workshops"
    ],

    # Tier 3 - Detailed Topics & Governance
    "Detailed Topics": [
        "convolutional neural network", "fuzzy logic", "matlab", "plc",
        "aws lambda", "mou", "sub-executive committee",
        "document coordinators", "bloom's taxonomy"
    ],
}


def detect_topic(text: str, topics_map: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    """
    Detect the most relevant topic for a given text using keyword matching.

    Args:
        text: input text to analyze.
        topics_map: optional dictionary mapping topic names to keywords
                    (defaults to TOPIC_KEYWORDS).

    Returns:
        The topic name (str) with the most keyword matches,
        or None if no topic is found.
    """
    if not text:
        return None

    text_lower = text.lower()
    topics = topics_map or TOPIC_KEYWORDS
    scores: Dict[str, int] = {}

    # Count keyword matches for each topic
    for topic_name, keywords in topics.items():
        match_count = sum(
            1 for kw in keywords
            if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text_lower)
        )
        if match_count > 0:
            scores[topic_name] = match_count

    if not scores:
        return None

    # Return topic with highest match count
    best_topic = max(scores.items(), key=lambda x: x[1])[0]
    return best_topic
