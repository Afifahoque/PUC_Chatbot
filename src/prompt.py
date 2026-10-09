# # src/prompt.py
# """
# Prompt management for Premier University RAG Chatbot.
# """

# from typing import List, Tuple
# # Use package-relative imports so Python finds these modules when importing src.prompt
# from .topics import detect_topic
# from .entities import extract_entities


# def get_system_prompt(user_question: str) -> str:
#     q = user_question.lower()
#     topic = detect_topic(q)

#     if topic == "Core Identity":
#         return (
#             "You are a friendly Premier University guide. "
#             "Answer the student's question about basic identity, admission, or university rules. "
#             "Use bullet points starting with '•' or a short paragraph. "
#             "Keep it clear and simple for new students. "
#             "{context}"
#         )
#     elif topic == "Curriculum & Wings":
#         return (
#             "You are a Premier University academic guide. "
#             "Explain courses, wings, workshops, or PEO/PO details. "
#             "Provide bullet points or short paragraphs. "
#             "Keep explanations easy for new students. "
#             "{context}"
#         )
#     elif topic == "Detailed Topics":
#         return (
#             "You are a Premier University advisor. "
#             "Answer questions about advanced topics, committees, or administrative rules. "
#             "Use concise bullet points or short paragraphs. "
#             "Keep it simple and actionable for students. "
#             "{context}"
#         )
#     else:
#         return (
#             "You are a Premier University chatbot. "
#             "Provide clear, concise answers suitable for new students. "
#             "Use bullet points starting with '•' or short paragraphs. "
#             "{context}"
#         )


# def inject_few_shot_examples(
#     messages: List[Tuple[str, str]],
#     examples: List[Tuple[str, str]]
# ) -> List[Tuple[str, str]]:
#     for q, a in examples:
#         lines = a.splitlines()
#         clean_lines = []
#         for line in lines:
#             line = line.strip()
#             if line and not line.startswith("•"):
#                 line = "• " + line.lstrip("-*1234567890. ")
#             clean_lines.append(line)
#         clean_answer = "\n".join(clean_lines)

#         messages.append(("human", q))
#         messages.append(("assistant", clean_answer))
#     return messages


# def build_prompt(user_question: str, context: str = "") -> str:
#     system_prompt = get_system_prompt(user_question)
#     prompt = system_prompt.format(context=context)
#     return prompt


# src/prompt.py
"""
Prompt management for Premier University RAG Chatbot.

Fully aligned with app.py:
- System prompts include {context} for dynamic entity/topic injection.
- Few-shot examples are injected in the correct "human"/"assistant" format.
- Safe bullet formatting for assistant responses.
"""

from typing import List, Tuple
from .topics import detect_topic
from .entities import extract_entities


def get_system_prompt(user_question: str) -> str:
    """
    Generate a system prompt based on detected topic and user question.
    {context} placeholder will be replaced in app.py with dynamic context.
    """
    q = user_question.lower()
    topic = detect_topic(q)

    if topic == "Core Identity":
        return (
            "You are a friendly Premier University guide. "
            "Answer the student's question about basic identity, admission, or university rules. "
            "Use bullet points starting with '•' or a short paragraph. "
            "Keep it clear and simple for new students. "
            "{context}"
        )
    elif topic == "Curriculum & Wings":
        return (
            "You are a Premier University academic guide. "
            "Explain courses, wings, workshops, or PEO/PO details. "
            "Provide bullet points or short paragraphs. "
            "Keep explanations easy for new students. "
            "{context}"
        )
    elif topic == "Detailed Topics":
        return (
            "You are a Premier University advisor. "
            "Answer questions about advanced topics, committees, or administrative rules. "
            "Use concise bullet points or short paragraphs. "
            "Keep it simple and actionable for students. "
            "{context}"
        )
    else:
        return (
            "You are a Premier University chatbot. "
            "Provide clear, concise answers suitable for new students. "
            "Use bullet points starting with '•' or short paragraphs. "
            "{context}"
        )


def inject_few_shot_examples(
    messages: List[Tuple[str, str]],
    examples: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """
    Inject few-shot Q&A examples into the message list for ChatPromptTemplate.

    Each example is added as ("human", question) followed by ("assistant", formatted answer).
    Ensures bullet points start with '•'.
    """
    for q, a in examples:
        lines = a.splitlines()
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("•"):
                line = "• " + line.lstrip("-*1234567890. ")
            clean_lines.append(line)
        clean_answer = "\n".join(clean_lines)

        messages.append(("human", q))
        messages.append(("assistant", clean_answer))
    return messages


def build_prompt(user_question: str, context: str = "") -> str:
    """
    Build the final system prompt for a single-turn question if needed.
    Not required for RAG chain since app.py handles messages/few-shot injection.
    """
    system_prompt = get_system_prompt(user_question)
    return system_prompt.format(context=context)
