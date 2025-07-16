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

# ------------------- INDEX HELPERS -------------------

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

# ------------------- EMBEDDING -------------------

def generate_embedding(text: str):
    response = bedrock_client.invoke_model(
        modelId=VECTOR_MODEL_ID,
        body=json.dumps({"inputText": text})
    )
    return json.loads(response["body"].read())["embedding"]

# ------------------- PDF PARSING -------------------

def extract_paragraphs_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Split into paragraphs
    paragraphs = [p.strip().replace('\n', ' ') for p in full_text.split("\n\n") if len(p.strip()) > 20]
    return paragraphs

# ------------------- EMBED & STORE CHUNK -------------------

def embed_and_store(text, pdf_path, vectors_list):
    if len(text.strip()) < 30:
        return  # Skip short content

    embedding = generate_embedding(text)

    # Truncate description to stay well within 2048-byte filterable metadata limit
    short_description = text.strip()[:512]

    vectors_list.append({
        "key": str(uuid.uuid4()),
        "data": {"float32": embedding},
        "metadata": {
            "description": short_description,
            "source_pdf": os.path.basename(pdf_path)
        }
    })

# ------------------- VECTOR INSERTION (with chunking) -------------------

def insert_descriptions_from_pdfs(pdf_paths, vector_bucket_name, index_name):
    vectors = []

    for pdf_path in pdf_paths:
        paragraphs = extract_paragraphs_from_pdf(pdf_path)
        ic(f"Extracted {len(paragraphs)} paragraphs from {pdf_path}")

        for desc in paragraphs:
            if len(desc) > 50000:
                # Chunk large text into ~4000-character pieces
                sentences = re.split(r'(?<=[.?!])\s+', desc)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 4000:
                        current_chunk += sentence + " "
                    else:
                        embed_and_store(current_chunk.strip(), pdf_path, vectors)
                        current_chunk = sentence + " "

                if current_chunk.strip():
                    embed_and_store(current_chunk.strip(), pdf_path, vectors)

            else:
                embed_and_store(desc, pdf_path, vectors)

    if vectors:
        s3_vector.put_vectors(
            vectorBucketName=vector_bucket_name,
            indexName=index_name,
            vectors=vectors
        )
        ic(f"Inserted {len(vectors)} descriptions into vector DB.")
    else:
        ic("No valid descriptions found to insert.")

# ------------------- VECTOR QUERY -------------------

def query_vector_store(query_text, top_k=2):
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
        metadata = result["metadata"]
        print("\n---")
        print("📄 Description:", metadata.get("description"))
        print("📁 Source:", metadata.get("source_pdf"))
        print("📏 Distance:", round(result["distance"], 4))

# ------------------- MAIN -------------------

if __name__ == "__main__":
    pdf_files = [
        "genai_book.pdf",
    ]

    list_vector_buckets()
    create_vector_index(INDEX_NAME)
    insert_descriptions_from_pdfs(pdf_files, VECTOR_BUCKET_NAME, INDEX_NAME)
    search_vector_store()
