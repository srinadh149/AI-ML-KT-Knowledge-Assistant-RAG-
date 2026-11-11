import os
import re

from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

KT_FILE = "AI_ML_KT.txt"


# ---------------------------------------------------------
# Load KT
# ---------------------------------------------------------

def load_knowledge():
    if not os.path.exists(KT_FILE):
        raise FileNotFoundError(
            f"{KT_FILE} was not found."
        )

    with open(KT_FILE, "r", encoding="utf-8") as file:
        return file.read()


# ---------------------------------------------------------
# Split KT into chunks
# ---------------------------------------------------------

def split_into_chunks(text):
    sections = re.split(
        r"\n(?=\d+[\.\)]\s)",
        text
    )

    chunks = []

    for section in sections:
        section = section.strip()

        if len(section) > 30:
            chunks.append(section)

    return chunks


# ---------------------------------------------------------
# Load and prepare knowledge base
# ---------------------------------------------------------

knowledge_text = load_knowledge()

chunks = split_into_chunks(knowledge_text)

vectorizer = TfidfVectorizer(
    stop_words="english"
)

chunk_vectors = vectorizer.fit_transform(chunks)


# ---------------------------------------------------------
# Retrieve relevant KT sections
# ---------------------------------------------------------

def retrieve_context(question, top_k=3):

    question_vector = vectorizer.transform(
        [question]
    )

    similarities = cosine_similarity(
        question_vector,
        chunk_vectors
    )[0]

    ranked_indexes = similarities.argsort()[::-1]

    results = []

    for index in ranked_indexes[:top_k]:

        if similarities[index] > 0:

            results.append(
                {
                    "text": chunks[index],
                    "score": similarities[index]
                }
            )

    return results


# ---------------------------------------------------------
# Generate answer from retrieved KT
# ---------------------------------------------------------

def generate_answer(question, results):

    if not results:
        return (
            "I could not find this information "
            "in the provided KT."
        )

    answer_parts = []

    for result in results:
        answer_parts.append(result["text"])

    return "\n\n".join(answer_parts)


# ---------------------------------------------------------
# Web Interface
# ---------------------------------------------------------

HTML = """
<!DOCTYPE html>

<html>

<head>

    <title>AI & ML KT RAG Assistant</title>

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 40px;
        }

        .container {
            max-width: 850px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
        }

        textarea {
            width: 100%;
            min-height: 120px;
            padding: 12px;
            font-size: 16px;
            box-sizing: border-box;
            margin-top: 15px;
        }

        button {
            margin-top: 15px;
            padding: 12px 25px;
            font-size: 16px;
            cursor: pointer;
        }

        .answer {
            margin-top: 25px;
            padding: 20px;
            background: #f0f4f8;
            border-radius: 8px;
            white-space: pre-wrap;
        }

        .source {
            margin-top: 15px;
            padding: 15px;
            background: #fafafa;
            border-left: 4px solid #555;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>AI & ML KT RAG Assistant</h1>

    <p>
        Ask a question about the
        Artificial Intelligence and Machine Learning
        Knowledge Transfer document.
    </p>


    <form method="POST">

        <textarea
            name="question"
            placeholder="Example: What is supervised learning?"
            required
        ></textarea>

        <button type="submit">
            Ask Question
        </button>

    </form>


    {% if answer %}

    <div class="answer">

        <h2>Retrieved Answer</h2>

        {{ answer }}

    </div>


    {% if sources %}

    <h2>Retrieved KT Sections</h2>

    {% for source in sources %}

    <div class="source">

        <strong>
            Similarity Score:
            {{ "%.3f"|format(source.score) }}
        </strong>

        <p>
            {{ source.text }}
        </p>

    </div>

    {% endfor %}

    {% endif %}

    {% endif %}


</div>

</body>

</html>
"""


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    answer = None
    sources = None

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        if question:

            sources = retrieve_context(
                question
            )

            answer = generate_answer(
                question,
                sources
            )

    return render_template_string(
        HTML,
        answer=answer,
        sources=sources
    )


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.route("/health")
def health():

    return {
        "status": "ok",
        "message": "AI ML RAG application is running"
    }


# ---------------------------------------------------------
# Run application
# ---------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )