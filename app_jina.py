from flask import Flask, request, jsonify, render_template
import pymupdf
import Stemmer
import bm25s
import requests
import numpy as np
from usearch.index import Index
import instructor
from pydantic import BaseModel, ValidationError
import os
import asyncio
import json
from dotenv import load_dotenv
from typing import List
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from werkzeug.utils import secure_filename
import openai

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = instructor.from_openai(openai.OpenAI())
jina_api_key = os.getenv("JINA_API_KEY")

# Pydantic response model
class ValidationResult(BaseModel):
    index: int
    excerpt: str
    answer: bool

class TfQuestions(BaseModel):
    questions: List[str]

# Helper functions to load files
def load_pdf(file_path, num_lines=15):
    pdf = pymupdf.open(file_path)
    text = []
    for page in pdf:
        for line in page.get_text("text").split("\n"):
            text.append(line)
    return ["\n".join(text[i:i+num_lines]) for i in range(0, len(text), num_lines)]

def load_text(file_path, num_lines=15):
    text = [line.replace("=", "").replace("-", "").replace("\n", "") for line in open(file_path, "r").readlines() if line.strip() != ""]
    text = ["\n".join(text[i:i+num_lines]) for i in range(0, len(text), num_lines)]
    return text

def embed(text, local=False, model=None):
    if not local:
        url = 'https://api.jina.ai/v1/embeddings'
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {jina_api_key}'
        }

        data = {
            "model": "jina-embeddings-v3",
            "task": "text-matching",
            "dimensions": 1024,
            "late_chunking": False,
            "embedding_type": "float",
            "input": [text]
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        try:
            embedding = response.json()['data'][0]['embedding']
            return embedding
        except Exception as e:
            raise e
    else:
        task_type = 'text-matching'
        return model.encode(
            [text],
            task=task_type,
            prompt_name=task_type,
        ).squeeze().tolist()

def generate_tf_questions(summary, n=20):
    prompt = f"""
    The goal is to validate whether the provided summary accurately aligns with the original document. 
    Generate a set of True/False questions based on the summary to capture the key claims in it. 
    All questions should be constructed so that their answers are 'True' if the summary is accurate, 
    because we aim to confirm alignment and not test comprehension. 

    Summary:
    {summary}
    
    return a list of {n} concise questions based on this summary.
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_model=TfQuestions
    )
    return response.questions

def initialize_index(text):
    stemmer = Stemmer.Stemmer('english')
    bm25 = bm25s.BM25(method="lucene")
    bm25.index(bm25s.tokenize(text, stemmer=stemmer))

    ind = Index(ndim=1024)
    embeddings = [np.array(embed(segment), dtype=np.float64) for segment in tqdm(text, desc="Generating embeddings")]
    embeddings = np.array(embeddings)
    indices = list(range(len(text)))
    ind.add(indices, embeddings)
    return bm25, ind

async def validate_question(question, index, bm25, text, top_k=6):
    question_embedding = np.array(embed(question.strip()))
    semantic = index.search(question_embedding, int(top_k / 2)).keys
    lexical = bm25.retrieve(bm25s.tokenize(question.strip()), k=3).documents[0]

    inds = set(semantic).union(lexical)
    context_segments = [(ind, text[ind]) for ind in inds]
    context = ', '.join([f"CONTEXT INDEX:{i}, CONTEXT SEGMENT: {segment}\n END OF SEGMENT {i}\n" for i, segment in context_segments])

    prompt = f"""
    Based on the following context:
    {context}
    Answer the question: {question}
    If True, specify the relevant index of the context segment and the exact text excerpt in that context that shows why you think the answer is True. If False, highlight the discrepancy between the question and the context, by providing the relevant index and excerpt. 
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=ValidationResult,
        messages=[{"role": "user", "content": prompt}]
    )
    return question, response

async def validate_questions(questions, index, bm25, text, top_k=6):
    tasks = [
        validate_question(question, index, bm25, text, top_k)
        for question in questions
    ]
    results = await async_tqdm.gather(
        *tasks,
        desc="Validating questions",
        total=len(questions)
    )
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
async def upload():
    if 'summary' not in request.files or 'detailed' not in request.files:
        return jsonify({"error": "Both files are required."}), 400

    summary_file = request.files['summary']
    detailed_file = request.files['detailed']

    if summary_file.filename == '' or detailed_file.filename == '':
        return jsonify({"error": "Files must have a valid name."}), 400

    summary_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(summary_file.filename))
    detailed_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(detailed_file.filename))

    summary_file.save(summary_path)
    detailed_file.save(detailed_path)

    # Process files
    results = await process_files(summary_path, detailed_path)

    return render_template('results.html', results=results)

async def process_files(summary_path, detailed_path):
    # Load detailed document
    if detailed_path.endswith(".pdf"):
        text = load_pdf(detailed_path)
    else:
        text = load_text(detailed_path)

    # Initialize indexes
    bm25, index = initialize_index(text)

    # Read summary content
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = f.read()

    # Generate True/False questions
    questions = generate_tf_questions(summary)

    # Validate questions
    validation_results = await validate_questions(questions, index, bm25, text)

    # Process results
    results = {"True": [], "False": []}
    for question, result in validation_results:
        try:
            structured_response = ValidationResult.model_validate(result)
            if structured_response.answer:
                results["True"].append({
                    "question": question,
                    "index": structured_response.index,
                    "support_text": structured_response.excerpt,
                    "original_context": text[structured_response.index]
                })
            else:
                results["False"].append({
                    "question": question,
                    "index": structured_response.index,
                    "misalignment_text": structured_response.excerpt,
                    "original_context": text[structured_response.index]
                })
        except ValidationError as e:
            results["False"].append({"question": question, "error": e.json()})

    return results

if __name__ == "__main__":
    app.run(debug=True)