#!/usr/bin/env python3
"""
Custom-batch embedding and ingestion script for Weaviate.
Generates embeddings in chunks of 100 to fit under Gemini API limits (1,000 requests/day).
Uses gemini-embedding-2 (3072 dimensions) for optimal vector search.
"""
import os
import sys
import time
import zipfile
import json
import httpx
import pandas as pd
from dotenv import load_dotenv
import weaviate.classes as wvc
from skinai.util.client import ClientManager

# ── Config ──────────────────────────────────────────────────
EMBED_BATCH_SIZE = 25  # Number of texts to embed in a single API call
MAX_RETRIES = 5         # Retries for embedding request on rate-limit
BACKOFF_BASE = 30       # Base backoff seconds on 429

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path, override=True)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY is not set in environment or .env file.")
    sys.exit(1)

print("=" * 60)
print("CLINIKALLY AGENTIC SKINCARE AI - CUSTOM BATCH INGESTION SCRIPT")
print("  (Using gemini-embedding-2 with pre-computed embeddings)")
print("=" * 60)

# Connect to Weaviate
cm = ClientManager()
print(f"Connecting to Weaviate at: {cm.wcd_url}...")

with cm.connect_to_client() as client:
    print("Connected successfully!")

    # ---------------------------------------------------------
    # 1. Setup SkincareProducts Collection
    # ---------------------------------------------------------
    product_collection_name = "SkincareProducts"
    if client.collections.exists(product_collection_name):
        print(f"Collection '{product_collection_name}' already exists. Recreating...")
        client.collections.delete(product_collection_name)

    print(f"Creating collection '{product_collection_name}'...")
    products_coll = client.collections.create(
        name=product_collection_name,
        properties=[
            wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="description", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="vendor", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="tags", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="url", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="image_src", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="price", data_type=wvc.config.DataType.NUMBER),
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_google_aistudio(
            model_id="gemini-embedding-2"
        ),
    )
    print(f"✅ Collection '{product_collection_name}' created.")

    # ---------------------------------------------------------
    # 2. Setup SkincareBlogs Collection
    # ---------------------------------------------------------
    blog_collection_name = "SkincareBlogs"
    if client.collections.exists(blog_collection_name):
        print(f"Collection '{blog_collection_name}' already exists. Recreating...")
        client.collections.delete(blog_collection_name)

    print(f"Creating collection '{blog_collection_name}'...")
    blogs_coll = client.collections.create(
        name=blog_collection_name,
        properties=[
            wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="author", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="created_at", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="updated_at", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="tags", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="summary", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="link", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_google_aistudio(
            model_id="gemini-embedding-2"
        ),
    )
    print(f"✅ Collection '{blog_collection_name}' created.")

    # ═════════════════════════════════════════════════════════
    # Helper: Fetch batch embeddings from Gemini API
    # ═════════════════════════════════════════════════════════
    def get_embeddings_batch(texts):
        """Get embeddings for a list of texts in a single batch API call.
        Extremely resilient: retries up to 100 times per call with capped exponential backoff.
        If it completely fails, sleeps 15 minutes and tries again."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={api_key}"
        
        # Clean and prepare requests
        cleaned_texts = []
        for t in texts:
            t_str = str(t).strip()
            # Ensure text is not empty and truncated to avoid exceeding limits
            if not t_str:
                t_str = " "
            cleaned_texts.append(t_str[:15000])

        payload = {
            "requests": [
                {
                    "model": "models/gemini-embedding-2",
                    "content": {"parts": [{"text": text}]}
                }
                for text in cleaned_texts
            ]
        }

        attempts = 0
        local_max_retries = 100
        while True:
            attempts += 1
            try:
                response = httpx.post(url, json=payload, timeout=90.0)
                if response.status_code == 200:
                    data = response.json()
                    return [emb["values"] for emb in data.get("embeddings", [])]
                elif response.status_code == 429:
                    # Calculate backoff: start at 30s, double each time, cap at 300s (5 minutes)
                    wait_time = min(300, 30 * (2 ** (min(5, attempts - 1))))
                    print(f"  ⏳ Rate limited (429). Attempt {attempts}/{local_max_retries}. Error response: {response.text[:200]}")
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  ⚠️ API error {response.status_code}. Attempt {attempts}/{local_max_retries}. Response: {response.text[:200]}")
                    wait_time = min(120, 10 * attempts)
                    time.sleep(wait_time)
            except Exception as e:
                print(f"  ⚠️ Connection error: {e}. Waiting 15s...")
                time.sleep(15)

            if attempts >= local_max_retries:
                print("❌ Failed after 100 attempts! Suspending execution for 15 minutes to let quota reset...")
                time.sleep(900)  # 15 minutes
                attempts = 0  # reset and try again

    # ---------------------------------------------------------
    # 3. Ingest SkincareProducts
    # ---------------------------------------------------------
    excel_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DermaGPT_Product_Database.xlsx"))
    print(f"\nReading products from {excel_path}...")
    df_products = pd.read_excel(excel_path)
    df_products = df_products.fillna("")
    total_products = len(df_products)
    print(f"Loaded {total_products} products.")

    # Prepare product objects and text for embedding
    product_objects = []
    product_texts = []
    for idx, row in df_products.iterrows():
        tags_str = str(row.get("Tags", ""))
        tags_list = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

        props = {
            "title": str(row.get("Title", "")),
            "description": str(row.get("D", "")),
            "vendor": str(row.get("Vendor", "")),
            "type": str(row.get("Type", "")),
            "tags": tags_list,
            "url": str(row.get("URL", "")),
            "category": str(row.get("Category", "")),
            "image_src": str(row.get("Image Src", "")),
        }
        try:
            props["price"] = float(row.get("Variant Price", 0.0))
        except Exception:
            props["price"] = 0.0
        
        product_objects.append(props)
        # Formulate text representation for vector search
        text_rep = f"Title: {props['title']} Description: {props['description']} Vendor: {props['vendor']} Type: {props['type']} Category: {props['category']}"
        product_texts.append(text_rep)

    # Fetch embeddings and insert into Weaviate
    print(f"\nGenerating embeddings and ingesting products in batches of {EMBED_BATCH_SIZE}...")
    inserted_products = 0
    for i in range(0, total_products, EMBED_BATCH_SIZE):
        chunk_props = product_objects[i : i + EMBED_BATCH_SIZE]
        chunk_texts = product_texts[i : i + EMBED_BATCH_SIZE]
        
        # 1. Fetch embeddings
        vectors = get_embeddings_batch(chunk_texts)
        
        # 2. Insert into Weaviate
        with products_coll.batch.dynamic() as batch:
            for props, vec in zip(chunk_props, vectors):
                batch.add_object(properties=props, vector=vec)
                
        inserted_products += len(chunk_props)
        print(f"  [{inserted_products}/{total_products}] products ingested")
        # Gentle pause between API calls to be friendly
        time.sleep(15.0)

    print(f"🎉 Ingested {inserted_products}/{total_products} products into '{product_collection_name}'!")

    # ---------------------------------------------------------
    # 4. Ingest SkincareBlogs
    # ---------------------------------------------------------
    zip_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DermaGPT_Blogs_with_Metadata.zip"))
    print(f"\nReading blogs from {zip_path}...")

    blog_objects = []
    blog_texts = []
    with zipfile.ZipFile(zip_path, "r") as z:
        metadata_files = [f for f in z.namelist() if f.endswith("metadata.json")]
        print(f"Found {len(metadata_files)} blog directories.")

        for meta_file in metadata_files:
            folder_path = os.path.dirname(meta_file)
            try:
                meta_bytes = z.read(meta_file)
                meta = json.loads(meta_bytes.decode("utf-8"))

                txt_file = f"{folder_path}/content_plain.txt"
                html_file = f"{folder_path}/content_clean.html"

                content = ""
                if txt_file in z.namelist():
                    content = z.read(txt_file).decode("utf-8", errors="ignore")
                elif html_file in z.namelist():
                    content = z.read(html_file).decode("utf-8", errors="ignore")

                tags_str = meta.get("tags", "")
                tags_list = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

                props = {
                    "title": str(meta.get("title", "")),
                    "author": str(meta.get("author", "")),
                    "created_at": str(meta.get("created_at", "")),
                    "updated_at": str(meta.get("updated_at", "")),
                    "tags": tags_list,
                    "summary": str(meta.get("summary", "")),
                    "link": str(meta.get("link", "")),
                    "content": content,
                }
                blog_objects.append(props)
                # Formulate text representation for vector search
                text_rep = f"Title: {props['title']} Summary: {props['summary']} Content: {props['content']}"
                blog_texts.append(text_rep)
            except Exception as e:
                print(f"⚠️ Error reading blog folder '{folder_path}': {e}")

    total_blogs = len(blog_objects)
    print(f"Loaded {total_blogs} blogs.")

    # Fetch embeddings and insert into Weaviate
    print(f"\nGenerating embeddings and ingesting blogs in batches of {EMBED_BATCH_SIZE}...")
    inserted_blogs = 0
    for i in range(0, total_blogs, EMBED_BATCH_SIZE):
        chunk_props = blog_objects[i : i + EMBED_BATCH_SIZE]
        chunk_texts = blog_texts[i : i + EMBED_BATCH_SIZE]
        
        # 1. Fetch embeddings
        vectors = get_embeddings_batch(chunk_texts)
        
        # 2. Insert into Weaviate
        with blogs_coll.batch.dynamic() as batch:
            for props, vec in zip(chunk_props, vectors):
                batch.add_object(properties=props, vector=vec)
                
        inserted_blogs += len(chunk_props)
        print(f"  [{inserted_blogs}/{total_blogs}] blogs ingested")
        # Gentle pause
        time.sleep(15.0)

    print(f"🎉 Ingested {inserted_blogs}/{total_blogs} blogs into '{blog_collection_name}'!")

print("\n" + "=" * 60)
print("ALL DATA INGESTION COMPLETED SUCCESSFULLY!")
print("=" * 60)
