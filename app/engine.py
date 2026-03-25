import json
import numpy as np
from typing import List
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from groq import Groq

from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from app.config import GROQ_API_KEY

# Load model ONCE
dense_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("API KEY:", GROQ_API_KEY)
def parse_document(path: str):
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline
            )
        }
    )
    return converter.convert(Path(path))

def clean_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
    return text

def chunk_document(doc_result):
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        max_tokens=512
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    chunks = list(chunker.chunk(doc_result.document))
    return [c.text.strip() for c in chunks if len(c.text.strip()) > 100]


def create_embeddings(texts: List[str]):
    return dense_model.encode(texts)


def retrieve_chunks(query, texts, embeddings, top_k):
    query_vec = dense_model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    idx = np.argsort(scores)[-top_k:][::-1]
    return [texts[i] for i in idx]



def generate_questions(context, difficulty, mcq, short, longq):
    client = Groq(api_key=GROQ_API_KEY)
    MODEL = "openai/gpt-oss-120b"

    prompt = f"""
You are an expert educator.

Difficulty Level: {difficulty}

Generate EXACTLY:
- {mcq} MCQs
- {short} Short Answer Questions
- {longq} Long Answer Questions

STRICT RULES FOR MCQs:
- Each MCQ must have 4 FULL descriptive options.
- Options must be complete meaningful statements.
- DO NOT use placeholders like "A", "B", "C", "D".
- Each option must contain real content.
- Mark the correct option clearly.

STRICT JSON FORMAT:

{{
  "mcq": [
    {{
      "question": "string",
      "options": [
        "Option 1: full sentence",
        "Option 2: full sentence",
        "Option 3: full sentence",
        "Option 4: full sentence"
      ],
      "answer": "Exact correct option text",
      "explanation": "string"
    }}
  ],
  "short_answer": [
    {{
      "question": "string",
      "answer": "string"
    }}
  ],
  "long_answer": [
    {{
      "question": "string",
      "answer": "string"
    }}
  ]
}}

ONLY return JSON.
NO markdown.
NO extra text.

Content:
{context}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content


def generate_assessment(file_path, difficulty, mcq, short, long):
    doc = parse_document(file_path)
    texts = chunk_document(doc)
    embeddings = create_embeddings(texts)

    basic = retrieve_chunks(
        "important key definitions concepts",
        texts, embeddings, 5
    )

    deep = retrieve_chunks(
        "analysis reasoning explanation implications",
        texts, embeddings, 3
    )

    context = "\n\n".join(basic + deep)

    result = generate_questions(context, difficulty, mcq, short, long)
    print(result)
    result = clean_json(result)
    return json.loads(result)