import json
import os
import re
import uuid

import boto3
import fitz  # PyMuPDF
from botocore.exceptions import ClientError
from icecream import ic

# ------------------- AWS CONFIG -------------------
AWS_ACCESS_KEY_ID = 'XXX'
AWS_SECRET_ACCESS_KEY = 'XXX'

VECTOR_BUCKET_NAME = "test-s3-vectordb"
INDEX_NAME = "test-s3-vectordb-index"
VECTOR_DIMENSION = 1024
VECTOR_MODEL_ID = "amazon.titan-embed-text-v2:0"

# ------------------- CLIENT SETUP -------------------
s3_vector = boto3.client("s3vectors")
bedrock_client = boto3.client("bedrock-runtime",
                              region_name="eu-central-1",
                              aws_access_key_id=AWS_ACCESS_KEY_ID,
                              aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# ------------------- HELPER FUNCTIONS -------------------

def list_vector_buckets():
    response = s3_vector.list_vector_buckets()
    buckets = [bucket["vectorBucketName"] for bucket in response["vectorBuckets"]]
    ic(buckets)

def get_index(index_name):
    try:
        s3_vector.get_index(
            indexName=index_name,
            vectorBucketName=VECTOR_BUCKET_NAME
        )
        return True
    except ClientError as exc:
        if "NotFoundException" in str(exc):
            ic("Index does not exist")
            return False
        else:
            raise

def create_vector_index(index_name=INDEX_NAME):
    if not get_index(index_name):
        s3_vector.create_index(
            indexName=index_name,
            vectorDimension=VECTOR_DIMENSION,
            distanceMetric="cosine",
            vectorBucketName=VECTOR_BUCKET_NAME
        )
        ic("Index created successfully.")
    else:
        ic("Index already exists.")

def generate_embedding(text: str):
    response = bedrock_client.invoke_model(
        modelId=VECTOR_MODEL_ID,
        body=json.dumps({"inputText": text})
    )
    return json.loads(response["body"].read())["embedding"]

# ------------------- PDF QUESTION PARSING -------------------

def extract_questions_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Use regex to extract lines ending with '?' as questions
    questions = re.findall(r"([^\n]{5,}?\?)", full_text)
    return [q.strip() for q in questions if len(q.strip()) > 10]

# ------------------- VECTOR INSERTION -------------------

def insert_questions_from_pdfs(pdf_paths, vector_bucket_name, index_name):
    vectors = []
    
    for pdf_path in pdf_paths:
        questions = extract_questions_from_pdf(pdf_path)
        ic(f"Extracted {len(questions)} questions from {pdf_path}")

        for question in questions:
            embedding = generate_embedding(question)
            vectors.append({
                "key": str(uuid.uuid4()),
                "data": {"float32": embedding},
                "metadata": {
                    "question": question,
                    "source_pdf": os.path.basename(pdf_path)
                }
            })

    if vectors:
        s3_vector.put_vectors(
            vectorBucketName=vector_bucket_name,
            indexName=index_name,
            vectors=vectors
        )
        ic(f"Inserted {len(vectors)} questions into vector DB.")
    else:
        ic("No questions found to insert.")

# ------------------- VECTOR QUERY -------------------

def query_vector_store(query_text, top_k=3):
    embedding = generate_embedding(query_text)
    response = s3_vector.query_vectors(
        vectorBucketName=VECTOR_BUCKET_NAME,
        indexName=INDEX_NAME,
        queryVector={"float32": embedding},
        topK=top_k,
        returnDistance=True,
        returnMetadata=True
    )
    return response["vectors"]

def search_vector_store():
    input_text = "How GenAI can add value for educators and students"
    vectors = query_vector_store(input_text)
    for result in vectors:
        ic(result["metadata"]["question"], result["distance"])

# ------------------- MAIN -------------------

if __name__ == "__main__":
    # Add your uploaded PDF file paths here
    pdf_files = [
        "genai_book.pdf",
    ]

    list_vector_buckets()
    create_vector_index(INDEX_NAME)
    insert_questions_from_pdfs(pdf_files, VECTOR_BUCKET_NAME, INDEX_NAME)
    search_vector_store()
