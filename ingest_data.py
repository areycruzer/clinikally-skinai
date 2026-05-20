#!/usr/bin/env python3
"""
Rate-limit-aware data ingestion for Weaviate with Google AI Studio embeddings.
Uses small fixed-size batches with pauses to stay under free-tier Gemini limits.
"""
import os
import sys
import time
import zipfile
import json
import pandas as pd
from dotenv import load_dotenv
import weaviate.classes as wvc
from skinai.util.client import ClientManager

# ── Config ──────────────────────────────────────────────────
BATCH_SIZE = 20        # Objects per batch (keep small for free tier)
PAUSE_BETWEEN = 2.0    # Seconds between batches
MAX_RETRIES = 5        # Retries per batch on rate-limit
BACKOFF_BASE = 30      # Base backoff seconds on 429

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path, override=True)

print("=" * 60)
print("CLINIKALLY AGENTIC SKINCARE AI - DATA INGESTION SCRIPT")
print("  (Rate-limit aware, small-batch mode)")
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
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_google_aistudio(model_id="gemini-embedding-001"),
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
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_google_aistudio(model_id="gemini-embedding-001"),
    )
    print(f"✅ Collection '{blog_collection_name}' created.")

    # ═════════════════════════════════════════════════════════
    # Helper: insert a small batch with retry on rate-limit
    # ═════════════════════════════════════════════════════════
    def insert_batch_with_retry(collection, objects_list, label=""):
        """Insert a list of property dicts into a collection, retrying on 429."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with collection.batch.fixed_size(batch_size=len(objects_list)) as batch:
                    for obj in objects_list:
                        batch.add_object(properties=obj)
                # Check for failed objects
                if collection.batch.failed_objects:
                    failed = collection.batch.failed_objects
                    err_msg = str(failed[0].message) if failed else ""
                    if "429" in err_msg or "quota" in err_msg.lower() or "rate" in err_msg.lower():
                        raise Exception(f"Rate limited: {err_msg[:200]}")
                    else:
                        print(f"  ⚠️ {len(failed)} objects failed: {err_msg[:200]}")
                        return len(objects_list) - len(failed)
                return len(objects_list)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    wait = BACKOFF_BASE * attempt
                    print(f"  ⏳ Rate limited (attempt {attempt}/{MAX_RETRIES}). Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  ❌ Batch error: {err_str[:300]}")
                    return 0
        print(f"  ❌ Failed after {MAX_RETRIES} retries.")
        return 0

    # ---------------------------------------------------------
    # 3. Ingest SkincareProducts from Excel
    # ---------------------------------------------------------
    excel_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DermaGPT_Product_Database.xlsx"))
    print(f"\nReading products from {excel_path}...")
    df_products = pd.read_excel(excel_path)
    df_products = df_products.fillna("")

    total_products = len(df_products)
    print(f"Ingesting {total_products} products in batches of {BATCH_SIZE}...")

    # Build all product objects first
    product_objects = []
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

    # Insert in small batches
    inserted_products = 0
    for i in range(0, len(product_objects), BATCH_SIZE):
        chunk = product_objects[i : i + BATCH_SIZE]
        count = insert_batch_with_retry(products_coll, chunk, label="products")
        inserted_products += count
        print(f"  [{inserted_products}/{total_products}] products ingested")
        time.sleep(PAUSE_BETWEEN)

    print(f"🎉 Ingested {inserted_products}/{total_products} products into '{product_collection_name}'!")

    # ---------------------------------------------------------
    # 4. Ingest SkincareBlogs from ZIP
    # ---------------------------------------------------------
    zip_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DermaGPT_Blogs_with_Metadata.zip"))
    print(f"\nReading blogs from {zip_path}...")

    blog_objects = []
    with zipfile.ZipFile(zip_path, "r") as z:
        metadata_files = [f for f in z.namelist() if f.endswith("metadata.json")]
        print(f"Found {len(metadata_files)} blogs in the zip archive.")

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
            except Exception as e:
                print(f"⚠️ Error reading blog folder '{folder_path}': {e}")

    total_blogs = len(blog_objects)
    print(f"Ingesting {total_blogs} blogs in batches of {BATCH_SIZE}...")

    inserted_blogs = 0
    for i in range(0, len(blog_objects), BATCH_SIZE):
        chunk = blog_objects[i : i + BATCH_SIZE]
        count = insert_batch_with_retry(blogs_coll, chunk, label="blogs")
        inserted_blogs += count
        print(f"  [{inserted_blogs}/{total_blogs}] blogs ingested")
        time.sleep(PAUSE_BETWEEN)

    print(f"🎉 Ingested {inserted_blogs}/{total_blogs} blogs into '{blog_collection_name}'!")

print("\n" + "=" * 60)
print("ALL DATA INGESTION COMPLETED!")
print("=" * 60)
