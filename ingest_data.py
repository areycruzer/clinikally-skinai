#!/usr/bin/env python3
import os
import sys
import zipfile
import json
import pandas as pd
from dotenv import load_dotenv
import weaviate.classes as wvc
from skinai.util.client import ClientManager

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path, override=True)

print("=" * 60)
print("CLINIKALLY AGENTIC SKINCARE AI - DATA INGESTION SCRIPT")
print("=" * 60)

# Connect to Weaviate
cm = ClientManager()
print(f"Connecting to Weaviate Cloud at: {cm.wcd_url}...")

with cm.connect_to_client() as client:
    print("Connected successfully!")
    
    # -------------------------------------------------------------
    # 1. Setup SkincareProducts Collection
    # -------------------------------------------------------------
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
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_google_gemini(model="text-embedding-004"),
    )
    print(f"✅ Collection '{product_collection_name}' created.")

    # -------------------------------------------------------------
    # 2. Setup SkincareBlogs Collection
    # -------------------------------------------------------------
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
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_google_gemini(model="text-embedding-004"),
    )
    print(f"✅ Collection '{blog_collection_name}' created.")

    # -------------------------------------------------------------
    # 3. Ingest SkincareProducts from Excel
    # -------------------------------------------------------------
    excel_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DermaGPT_Product_Database.xlsx"))
    print(f"\nReading products from {excel_path}...")
    df_products = pd.read_excel(excel_path)
    
    # Fill nan values
    df_products = df_products.fillna("")
    
    print(f"Ingesting {len(df_products)} products in batches...")
    
    # We will use Weaviate batching
    with products_coll.batch.dynamic() as batch:
        for idx, row in df_products.iterrows():
            # Clean and split tags
            tags_str = str(row.get("Tags", ""))
            tags_list = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
            
            properties = {
                "title": str(row.get("Title", "")),
                "description": str(row.get("D", "")),
                "vendor": str(row.get("Vendor", "")),
                "type": str(row.get("Type", "")),
                "tags": tags_list,
                "url": str(row.get("URL", "")),
                "category": str(row.get("Category", "")),
                "image_src": str(row.get("Image Src", "")),
            }
            # Handle float price
            try:
                properties["price"] = float(row.get("Variant Price", 0.0))
            except Exception:
                properties["price"] = 0.0
                
            batch.add_object(properties=properties)
            
            if (idx + 1) % 500 == 0:
                print(f"  Ingested {idx + 1} products...")
                
    print(f"🎉 Successfully ingested products into '{product_collection_name}'!")

    # -------------------------------------------------------------
    # 4. Ingest SkincareBlogs from ZIP
    # -------------------------------------------------------------
    zip_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DermaGPT_Blogs_with_Metadata.zip"))
    print(f"\nReading blogs from {zip_path}...")
    
    blog_count = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        # Find all metadata.json files to know where the blogs are
        metadata_files = [f for f in z.namelist() if f.endswith("metadata.json")]
        print(f"Found {len(metadata_files)} blogs in the zip archive.")
        
        print("Ingesting blogs in batches...")
        with blogs_coll.batch.dynamic() as batch:
            for idx, meta_file in enumerate(metadata_files):
                # The folder path is the parent of metadata.json
                folder_path = os.path.dirname(meta_file)
                
                try:
                    # Read metadata
                    meta_bytes = z.read(meta_file)
                    meta = json.loads(meta_bytes.decode("utf-8"))
                    
                    # Read plain content if it exists, otherwise clean html
                    txt_file = f"{folder_path}/content_plain.txt"
                    html_file = f"{folder_path}/content_clean.html"
                    
                    content = ""
                    if txt_file in z.namelist():
                        content = z.read(txt_file).decode("utf-8", errors="ignore")
                    elif html_file in z.namelist():
                        content = z.read(html_file).decode("utf-8", errors="ignore")
                    
                    # Clean and split tags
                    tags_str = meta.get("tags", "")
                    tags_list = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
                    
                    properties = {
                        "title": str(meta.get("title", "")),
                        "author": str(meta.get("author", "")),
                        "created_at": str(meta.get("created_at", "")),
                        "updated_at": str(meta.get("updated_at", "")),
                        "tags": tags_list,
                        "summary": str(meta.get("summary", "")),
                        "link": str(meta.get("link", "")),
                        "content": content,
                    }
                    
                    batch.add_object(properties=properties)
                    blog_count += 1
                    
                    if blog_count % 500 == 0:
                        print(f"  Ingested {blog_count} blogs...")
                        
                except Exception as e:
                    print(f"⚠️ Error reading blog folder '{folder_path}': {e}")
                    
    print(f"🎉 Successfully ingested {blog_count} blogs into '{blog_collection_name}'!")

print("\n" + "=" * 60)
print("ALL DATA INGESTION COMPLETED SUCCESSFULLY!")
print("=" * 60)
