
import json
import os
import re
import uuid

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from icecream import ic

# AWS Configuration
AWS_ACCESS_KEY_ID = 'XXX'
AWS_SECRET_ACCESS_KEY = 'XXX'
VECTOR_BUCKET_NAME = "vikash_test-s3-vectordb"
INDEX_NAME = "vikash_test-s3-vectordb-index"
VECTOR_DIMENSION = 1024
VECTOR_MODEL_ID = "amazon.titan-embed-text-v2:0"

s3_vector = boto3.client("s3vectors")
bedrock_client = boto3.client("bedrock-runtime",
                              region_name="eu-central-1",
                              aws_access_key_id=AWS_ACCESS_KEY_ID,
                              aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# ------------- Embedding -------------
def generate_embedding(text: str):
    response = bedrock_client.invoke_model(
        modelId=VECTOR_MODEL_ID,
        body=json.dumps({"inputText": text})
    )
    return json.loads(response["body"].read())["embedding"]

# ------------- Embed & Store -------------
def embed_and_store(text, source_id, vectors_list):
    if len(text.strip()) < 30:
        return
    embedding = generate_embedding(text)
    short_description = text.strip()[:512]
    vectors_list.append({
        "key": str(uuid.uuid4()),
        "data": {"float32": embedding},
        "metadata": {
            "description": short_description,
            "source_csv": source_id
        }
    })

# ------------- Parse CSV Rows -------------
def extract_rows_from_csv(csv_path, text_columns):
    df = pd.read_csv(csv_path)
    rows = []

    for _, row in df.iterrows():
        combined = ' | '.join(str(row[col]) for col in text_columns if pd.notna(row[col]))
        if len(combined.strip()) > 30:
            rows.append(combined.strip())

    return rows

# ------------- Insert into Vector DB -------------
def insert_descriptions_from_csv(csv_path, vector_bucket_name, index_name, text_columns):
    vectors = []
    rows = extract_rows_from_csv(csv_path, text_columns)
    ic(f"Extracted {len(rows)} rows from {csv_path}")

    for row_text in rows:
        if len(row_text) > 50000:
            # Chunk if needed
            sentences = re.split(r'(?<=[.?!])\s+', row_text)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 4000:
                    current_chunk += sentence + " "
                else:
                    embed_and_store(current_chunk.strip(), csv_path, vectors)
                    current_chunk = sentence + " "
            if current_chunk.strip():
                embed_and_store(current_chunk.strip(), csv_path, vectors)
        else:
            embed_and_store(row_text, csv_path, vectors)

    if vectors:
        s3_vector.put_vectors(
            vectorBucketName=vector_bucket_name,
            indexName=index_name,
            vectors=vectors
        )
        ic(f"Inserted {len(vectors)} rows from CSV.")
    else:
        ic("No valid data to insert.")

# ------------- Query Function -------------
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
    query = "Hazardous materials shipped to Germany"
    results = query_vector_store(query)
    for result in results:
        metadata = result["metadata"]
        print("\n---")
        print("📄 Description:", metadata.get("description"))
        print("📁 Source:", metadata.get("source_csv"))
        print("📏 Distance:", round(result["distance"], 4))

# ------------- Index & Bucket Setup -------------
def create_vector_index(index_name=INDEX_NAME):
    try:
        s3_vector.get_index(indexName=index_name, vectorBucketName=VECTOR_BUCKET_NAME)
        ic("Index already exists.")
    except ClientError as e:
        if "NotFoundException" in str(e):
            s3_vector.create_index(
                indexName=index_name,
                vectorDimension=VECTOR_DIMENSION,
                distanceMetric="cosine",
                vectorBucketName=VECTOR_BUCKET_NAME
            )
            ic("Index created.")
        else:
            raise

# ------------- MAIN RUN -------------
if __name__ == "__main__":
    csv_file = Branded add on label.csv"  # <-- replace with actual path
    text_columns = [
        "MATNR_DESCRIPTION", 
        "SHIP_from_country", 
        "SHIP_to_country", 
        "HazardousStatus"
    ]

    create_vector_index()
    insert_descriptions_from_csv(csv_file, VECTOR_BUCKET_NAME, INDEX_NAME, text_columns)
    search_vector_store()
