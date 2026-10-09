# app.py
"""
Flask application for Premier University Chatbot with debug logs.

- Prints retrieved documents for debugging
- Handles LLM / Pinecone errors gracefully
"""

import os
import re
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
from markupsafe import escape

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Local helpers
from src.helper import download_hugging_face_embeddings
from src.prompt import get_system_prompt
from src.entities import extract_entities
from src.topics import detect_topic

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# ----------------------------
# Config & logging
# ----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
INDEX_NAME = os.environ.get("PINECONE_INDEX", "puc-chatbot")

if not PINECONE_API_KEY:
    raise SystemExit("PINECONE_API_KEY missing in .env — required to run the app.")

# ----------------------------
# Initialize Pinecone
# ----------------------------
logger.info("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(INDEX_NAME):
    logger.info(f"Index '{INDEX_NAME}' missing! Run store_index.py first.")
    raise SystemExit("Pinecone index missing! Upload vectors first.")

index = pc.Index(INDEX_NAME)

# ----------------------------
# Load embeddings & retriever
# ----------------------------
logger.info("Loading HuggingFace embeddings...")
embeddings = download_hugging_face_embeddings()

logger.info("Connecting to PineconeVectorStore (existing index)...")
try:
    docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    logger.info("Retriever ready.")
except Exception as e:
    logger.exception("Failed to create retriever: %s", e)
    retriever = None

# ----------------------------
# LLM Model
# ----------------------------
if LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY missing — set LLM_PROVIDER=groq to use Groq instead.")
    chatModel = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        api_key=OPENAI_API_KEY,
    )
elif LLM_PROVIDER == "groq":
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY missing — create a Groq API key and add it to .env.")
    chatModel = ChatGroq(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=GROQ_API_KEY,
    )
else:
    raise SystemExit("Unsupported LLM_PROVIDER. Choose 'openai' or 'groq'.")

# ----------------------------
# Memory
# ----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_input", return_messages=False)

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)


def format_answer_to_html(raw_answer: str) -> str:
    if raw_answer is None:
        return "<div></div>"
    ans = str(raw_answer).strip().replace("\r\n", "\n").replace("\r", "\n")
    if "•" in ans:
        items = [escape(ln.lstrip("•").strip()) if ln.startswith("•") else escape(ln) for ln in ans.split("\n") if ln.strip()]
        html = "<ul style='margin:0;padding-left:20px;'>" + "".join(f"<li>{it}</li>" for it in items) + "</ul>"
        return html
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", ans) if p.strip()]
    if len(paragraphs) > 1:
        return "".join(f"<p>{escape(p)}</p>" for p in paragraphs)
    return f"<div>{escape(ans)}</div>"


@app.route("/")
def index_route():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    try:
        user_input = request.form.get("msg", "").strip()
        if not user_input:
            return jsonify({"html": "<div>No input provided</div>", "entities": [], "topic": None})

        # 1) Entities & topic
        try:
            entities = extract_entities(user_input)
        except Exception as e:
            logger.debug("Entity extraction failed: %s", e)
            entities = []

        try:
            topic = detect_topic(user_input)
        except Exception as e:
            logger.debug("Topic detection failed: %s", e)
            topic = None

        # Save user input into memory
        try:
            memory.save_context({"user_input": user_input}, {"output": ""})
        except Exception:
            pass

        # 2) Retrieve documents
        retrieved_docs = []
        if retriever:
            try:
                retrieved_docs = retriever.get_relevant_documents(user_input)
                logger.info(f"Retrieved {len(retrieved_docs)} docs for input: {user_input}")
            except Exception as e:
                logger.exception("Retriever failed: %s", e)

        # Debug fallback: if no docs retrieved, return static info
        if not retrieved_docs:
            answer = "Premier University is a reputed institution offering B.Sc in CSE and various workshops."
            return jsonify({"html": format_answer_to_html(answer), "entities": entities, "topic": topic})

        # 3) Build prompt template
        guard = "Answer only the current user question. Do not assume past topics unless explicitly referenced."
        base_system = get_system_prompt(user_input).replace("{context}", "").strip()
        few_shot_text = "Q: What is Premier University?\nA: • Premier University is a reputed institution.\n• Offers B.Sc in CSE.\n\nQ: What workshops are available?\nA: • AI and ML bootcamps\n• Networking workshops"
        template_text = f"{guard} {base_system}\n\n{few_shot_text}\n\nContext:\n{{context}}\n\nQuestion:\n{{input}}\n\nAnswer:"

        prompt = PromptTemplate(template=template_text, input_variables=["context", "input"])

        # 4) Chains
        question_answer_chain = create_stuff_documents_chain(llm=chatModel, prompt=prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": user_input})

        # Extract answer
        answer = ""
        if isinstance(response, dict):
            answer = response.get("answer") or response.get("output_text") or ""
        elif isinstance(response, str):
            answer = response
        else:
            answer = str(response)

        # Save assistant output to memory
        try:
            memory.save_context({"user_input": user_input}, {"output": answer})
        except Exception:
            pass

        return jsonify({"html": format_answer_to_html(answer), "entities": entities, "topic": topic})

    except Exception as exc:
        logger.exception("Unhandled exception in /get: %s", exc)
        return jsonify({"html": "<div>Sorry, an internal error occurred.</div>", "entities": [], "topic": None}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
