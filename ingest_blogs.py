#!/usr/bin/env python3
"""Ingest blogs into Weaviate - fixed version with smaller batches and content truncation."""
import os
import sys
import zipfile
import json
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import weaviate.classes as wvc
from skinai.util.client import ClientManager

# Load .env
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path, override=True)

MAX_CONTENT_LENGTH = 8000  # Truncate blog content to avoid vectorizer timeouts

print("=" * 60)
print("BLOG INGESTION SCRIPT (FIXED)")
print("=" * 60)

cm = ClientManager()
print(f"Connecting to Weaviate...")

with cm.connect_to_client() as client:
    print("Connected!")

    blog_collection_name = "SkincareBlogs"

    # Delete and recreate
    if client.collections.exists(blog_collection_name):
        print(f"Deleting existing '{blog_collection_name}' (had only 48 objects)...")
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
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_weaviate(),
    )
    print(f"✅ Collection created.")

    # Ingest blogs
    zip_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DermaGPT_Blogs_with_Metadata.zip"))
    print(f"\nReading blogs from ZIP...")

    blog_count = 0
    skipped = 0
    errors = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        # Find metadata.json files, SKIP __MACOSX junk
        metadata_files = [
            f for f in z.namelist()
            if f.endswith("metadata.json") and not f.startswith("__MACOSX")
        ]
        total = len(metadata_files)
        print(f"Found {total} valid blogs (excluded __MACOSX).")

        # Use fixed_size batches of 25 to avoid timeout
        with blogs_coll.batch.fixed_size(batch_size=25) as batch:
            for idx, meta_file in enumerate(metadata_files):
                folder_path = os.path.dirname(meta_file)
                try:
                    meta_bytes = z.read(meta_file)
                    meta = json.loads(meta_bytes.decode("utf-8", errors="replace"))

                    # Read plain text content
                    txt_file = f"{folder_path}/content_plain.txt"
                    html_file = f"{folder_path}/content_clean.html"

                    content = ""
                    if txt_file in z.namelist():
                        content = z.read(txt_file).decode("utf-8", errors="replace")
                    elif html_file in z.namelist():
                        content = z.read(html_file).decode("utf-8", errors="replace")

                    # Truncate oversized content
                    if len(content) > MAX_CONTENT_LENGTH:
                        content = content[:MAX_CONTENT_LENGTH]

                    # Skip empty blogs
                    title = str(meta.get("title", "")).strip()
                    if not title and not content:
                        skipped += 1
                        continue

                    tags_str = meta.get("tags", "")
                    tags_list = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

                    properties = {
                        "title": title,
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

                    if blog_count % 100 == 0:
                        print(f"  Progress: {blog_count}/{total} blogs ingested...")

                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"  ⚠️ Error on '{folder_path}': {e}")

    print(f"\n{'=' * 60}")
    print(f"BLOG INGESTION COMPLETE!")
    print(f"  ✅ Ingested: {blog_count}")
    print(f"  ⏭️  Skipped (empty): {skipped}")
    print(f"  ⚠️  Errors: {errors}")
    print(f"{'=' * 60}")

    # Verify
    agg = blogs_coll.aggregate.over_all(total_count=True)
    print(f"\nVerification: '{blog_collection_name}' now has {agg.total_count} objects.")
