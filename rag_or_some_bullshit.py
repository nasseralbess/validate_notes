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
import openai
from dotenv import load_dotenv
from typing import List
from tqdm import tqdm
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = instructor.from_openai(openai.OpenAI())

# Pydantic response model
class ValidationResult(BaseModel):
    index: int
    excerpt: str
    answer: bool

class TfQuestions(BaseModel):
    questions: List[str]

# Load PDF and extract text
def load_pdf(file_path, num_lines=15):
    pdf = pymupdf.open(file_path)
    text = []
    for page in pdf:
        for line in page.get_text("text").split("\n"):
            text.append(line)
    return ["\n".join(text[i:i+num_lines]) for i in range(0, len(text), num_lines)]

def load_text(file_path, num_lines=15):
    text = [line.replace("=","").replace("-","").replace("\n","") for line in open(file_path,"r").readlines() if line.strip() != ""]
    text = ["\n".join(text[i:i+num_lines]) for i in range(0,len(text), num_lines)]
    return text

def embed(text):
    url = 'http://37.27.141.74:7997/embeddings'
    auth = os.getenv("EMBEDDING_API_KEY")

    body = {
        "model": "intfloat/multilingual-e5-large-instruct",
        "encoding_format": "float",
        "user": "string",
        "dimensions": 0,
        "input": [text],
        "modality": "text"
    }
    response = requests.post(url, json=body, headers={'Authorization': f'Bearer {auth}'})
    return response.json()['data'][0]['embedding']

def embed_bulk(text):
    url = 'http://37.27.141.74:7997/embeddings'
    auth = os.getenv("EMBEDDING_API_KEY")

    body = {
        "model": "intfloat/multilingual-e5-large-instruct",
        "encoding_format": "float",
        "user": "string",
        "dimensions": 0,
        "input": text,
        "modality": "text"
    }
    response = requests.post(url, json=body, headers={'Authorization': f'Bearer {auth}'})
    embeddings, indices = [list(response.json()['data'][i].values())[1] for i in range(len(response.json()['data']))], [list(response.json()['data'][i].values())[2] for i in range(len(response.json()['data']))]
    embeddings = np.array([np.array(embedding, dtype=np.float64) for embedding in embeddings])
    return embeddings, indices

def generate_tf_questions(summary, n=20):
    prompt = f"""
    The goal is to validate whether the provided summary accurately aligns with the original document. 
    Generate a set of True/False questions based on the summary capture the key claims in it. 
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
    
    # Extract questions from the response
    return response.questions
# Initialize models and index
def initialize_index(text):
    # bm25 = bm25s.BM25.load("bm25_medic")
    # ind = Index(ndim=1024)
    # ind.load("index_medic.usearch")
    stemmer = Stemmer.Stemmer('english')
    bm25 = bm25s.BM25(method="lucene")
    bm25.index(bm25s.tokenize(text, stemmer=stemmer))
    # bm25.save(f"bm25_medic")

    ind = Index(ndim=1024)
    # embeddings, indices = embed_bulk(text)
    embeddings = []
    # np.array([np.array(embed(segment), dtype=np.float64) for segment in text])
    for segment in tqdm(text):
        embeddings.append(np.array(embed(segment), dtype=np.float64))
    embeddings = np.array(embeddings)   
    indices = list(range(len(text)))
    ind.add(indices, embeddings)
    # ind.save(f"index_medic.usearch")
    return bm25, ind

# Validate True/False questions
async def validate_question(question, index, bm25, text, top_k=6):
    print(question)
    question_embedding = np.array(embed(question.strip()))
    semantic = index.search(question_embedding, int(top_k / 2)).keys
    lexical = bm25.retrieve(bm25s.tokenize(question.strip()), k=3).documents[0]

    inds = set(semantic).union(lexical)
    # inds = lexical
    print(inds)
    context_segments = [(ind, text[ind]) for ind in inds]
    # print("QUESTION\n----------------------------------\n", question, "\n----------------------------------\n", context_segments, "\n----------------------------------\n\n")

    prompt = f"""
    Based on the following context:
    {', '.join([f"[{i}] {segment}" for i, segment in context_segments])}

    Answer the question: {question}
    If True, specify the relevant index of the context segment and the exact text excerpt in that context that shows why you think the answer is True. If False, highlight the discrepancy between the question and the context, by providing the relevant index and excerpt. 
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=ValidationResult,
        messages=[{"role": "user", "content": prompt}]
    )
    # Clean invalid JSON characters
    return question,response

async def validate_questions(questions, index, bm25, text, top_k=6):
    tasks = [
        validate_question(question, index, bm25, text, top_k)
        for question in questions
    ]
    return await asyncio.gather(*tasks)


# Main workflow
def main(summary_path, detailed_text_path):
    # Load and preprocess text
    if detailed_text_path.endswith(".pdf"):
        text = load_pdf(detailed_text_path)
    else:
        text = load_text(detailed_text_path)

    # with open("text.txt", "r", encoding="utf-8") as f:
    #     text = f.read().split("|")
    bm25, index = initialize_index(text)

    summary = open(summary_path, "r", encoding="utf-8").read()
    # Generate and validate questions
    questions = generate_tf_questions(summary)
    with open("questions_medic.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(questions))
    # with open("questions.txt", "r", encoding="utf-8") as f:
    #     questions = f.read().split("\n")
    # questions = ['Is the AMIR approach designed to address the challenges faced in Arabic Information Retrieval (AIR)?', 'Do existing Arabic stemming tools suffer from high error rates due to incomplete affix removal?', 'Does the AMIR system utilize a dictionary to validate root extraction and improve retrieval accuracy?', 'Was the performance of AMIR evaluated against LUCENE and FARASA stemmers using the EveTAR dataset?', 'Is there a suggestion in the study for extending the AMIR approach to handle informal Arabic forms?']
    validation_results = asyncio.run(validate_questions(questions, index, bm25, text))
    with open("validation_results_medic.txt", "w", encoding="utf-8") as f:
        f.write("\n".join([str(result) for result in validation_results]))

    for question, result in validation_results:
        try:
            structured_response = ValidationResult.model_validate(result)
            print("Question:", question)
            print(structured_response.model_dump_json(indent=2))
            if structured_response.index is not None:
                print(f"Context from index [{structured_response.index}]: {text[structured_response.index]}\n")
        except ValidationError as e:
            print(f"Validation error: {e.json()}")

if __name__ == "__main__":
    summary_path = "AI-generate_discharge.txt"
    detailed_text_path = "original_note.txt"
    main(summary_path, detailed_text_path)
    # summary = open("summary.txt", "r", encoding="utf-8").read()
    # print(generate_tf_questions(summary, 5))

