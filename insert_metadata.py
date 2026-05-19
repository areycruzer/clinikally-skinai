import os
import asyncio
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(override=True)

import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from skinai.util.client import ClientManager
from skinai.preprocessing.collection import (
    _evaluate_index_properties,
    _find_vectorisers,
    preprocessed_collection_exists_async,
    delete_preprocessed_collection_async
)

# Set Weaviate variables
os.environ["WEAVIATE_IS_LOCAL"] = "False"

async def main():
    print("=" * 60)
    print("MANUAL METADATA INSERTION SCRIPT")
    print("=" * 60)
    
    cm = ClientManager()
    print("Connecting to Weaviate...")
    
    async with cm.connect_to_async_client() as client:
        print("Connected!")
        
        # Ensure ELYSIA_METADATA__ collection exists
        metadata_name = "ELYSIA_METADATA__"
        if not await client.collections.exists(metadata_name):
            print(f"Creating metadata collection '{metadata_name}'...")
            await client.collections.create(
                name=metadata_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="name", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="length", data_type=DataType.NUMBER),
                    Property(name="summary", data_type=DataType.TEXT),
                    Property(
                        name="index_properties",
                        data_type=DataType.OBJECT,
                        nested_properties=[
                            Property(name="isNullIndexed", data_type=DataType.BOOL),
                            Property(name="isLengthIndexed", data_type=DataType.BOOL),
                            Property(name="isTimestampIndexed", data_type=DataType.BOOL),
                        ],
                    ),
                    Property(
                        name="named_vectors",
                        data_type=DataType.OBJECT_ARRAY,
                        nested_properties=[
                            Property(name="name", data_type=DataType.TEXT),
                            Property(name="vectorizer", data_type=DataType.TEXT),
                            Property(name="model", data_type=DataType.TEXT),
                            Property(name="source_properties", data_type=DataType.TEXT_ARRAY),
                            Property(name="enabled", data_type=DataType.BOOL),
                            Property(name="description", data_type=DataType.TEXT),
                        ],
                    ),
                    Property(
                        name="vectorizer",
                        data_type=DataType.OBJECT,
                        nested_properties=[
                            Property(name="vectorizer", data_type=DataType.TEXT),
                            Property(name="model", data_type=DataType.TEXT),
                        ],
                    ),
                    Property(
                        name="fields",
                        data_type=DataType.OBJECT_ARRAY,
                        nested_properties=[
                            Property(name="name", data_type=DataType.TEXT),
                            Property(name="type", data_type=DataType.TEXT),
                            Property(name="description", data_type=DataType.TEXT),
                            Property(name="range", data_type=DataType.NUMBER_ARRAY),
                            Property(name="date_range", data_type=DataType.DATE_ARRAY),
                            Property(
                                name="groups",
                                data_type=DataType.OBJECT_ARRAY,
                                nested_properties=[
                                    Property(name="value", data_type=DataType.TEXT),
                                    Property(name="count", data_type=DataType.INT),
                                ],
                            ),
                            Property(name="date_median", data_type=DataType.DATE),
                            Property(name="mean", data_type=DataType.NUMBER),
                        ],
                    ),
                ],
                inverted_index_config=Configure.inverted_index(index_null_state=True),
            )
            print("✅ Metadata collection created.")
        
        metadata_coll = client.collections.get(metadata_name)

        # ---------------------------------------------------------------------
        # 1. PREPROCESS SKINCARE PRODUCTS
        # ---------------------------------------------------------------------
        print("\n--- Processing 'SkincareProducts' ---")
        prod_coll_name = "SkincareProducts"
        if not await client.collections.exists(prod_coll_name):
            print(f"❌ Collection '{prod_coll_name}' does not exist! Skip.")
        else:
            prod_coll = client.collections.get(prod_coll_name)
            
            # Fetch count
            agg = await prod_coll.aggregate.over_all(total_count=True)
            len_prod = agg.total_count
            print(f"  Count: {len_prod}")
            
            # Index properties & vectorizers from live schema
            idx_props = await _evaluate_index_properties(prod_coll)
            named_vecs, vec = await _find_vectorisers(prod_coll)
            
            # Define fields
            fields = [
                {"name": "title", "type": "text", "description": "The product title or name.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "description", "type": "text", "description": "Detailed description of the skincare product, benefits, and ingredients.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "vendor", "type": "text", "description": "The brand or manufacturer of the product.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "type", "type": "text", "description": "The categorization type of product (e.g. Cleanser, Sunscreen, Cream).", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "tags", "type": "text[]", "description": "Comma-separated search tags, ingredients, and stock location statuses.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "url", "type": "text", "description": "Website product URL link.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "category", "type": "text", "description": "Hierarchical e-commerce category path.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "image_src", "type": "text", "description": "The hosted image source URL.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "price", "type": "float", "description": "Product price in INR.", "range": [10.0, 20000.0], "date_range": None, "groups": None, "date_median": None, "mean": 1500.0},
            ]
            
            # Define mappings
            mappings = {
                "product": {
                    "name": "title",
                    "description": "description",
                    "price": "price",
                    "category": "category",
                    "subcategory": "type",
                    "tags": "tags",
                    "url": "url",
                    "image": "image_src",
                    "brand": "vendor",
                },
                "document": {
                    "title": "title",
                    "author": "vendor",
                    "date": "",
                    "content": "description",
                    "category": "category",
                },
                "generic": {
                    "title": "title",
                    "subtitle": "type",
                    "content": "description",
                    "url": "url",
                    "author": "vendor",
                    "tags": "tags",
                    "category": "category",
                },
                "table": {
                    "title": "title",
                    "description": "description",
                    "vendor": "vendor",
                    "type": "type",
                    "tags": "tags",
                    "url": "url",
                    "category": "category",
                    "image_src": "image_src",
                    "price": "price"
                }
            }
            
            prod_meta = {
                "name": prod_coll_name,
                "length": len_prod,
                "summary": "A dataset of 2,999 skincare products from the Clinikally database. It contains titles, detailed descriptions, vendors (brands), product types, search tags, URLs, categories, image source links, and pricing in INR.",
                "index_properties": idx_props,
                "named_vectors": named_vecs,
                "vectorizer": vec,
                "fields": fields,
                "mappings": mappings
            }
            
            # Delete if exists in metadata
            if await preprocessed_collection_exists_async(prod_coll_name, cm):
                print(f"  Removing old preprocessed metadata for {prod_coll_name}...")
                await delete_preprocessed_collection_async(prod_coll_name, cm)
            
            # Write to metadata
            await metadata_coll.data.insert(prod_meta)
            print(f"  ✅ Preprocessed metadata for '{prod_coll_name}' written successfully!")

        # ---------------------------------------------------------------------
        # 2. PREPROCESS SKINCARE BLOGS
        # ---------------------------------------------------------------------
        print("\n--- Processing 'SkincareBlogs' ---")
        blog_coll_name = "SkincareBlogs"
        if not await client.collections.exists(blog_coll_name):
            print(f"❌ Collection '{blog_coll_name}' does not exist! Skip.")
        else:
            blog_coll = client.collections.get(blog_coll_name)
            
            # Fetch count
            agg = await blog_coll.aggregate.over_all(total_count=True)
            len_blog = agg.total_count
            print(f"  Count: {len_blog}")
            
            # Index properties & vectorizers from live schema
            idx_props = await _evaluate_index_properties(blog_coll)
            named_vecs, vec = await _find_vectorisers(blog_coll)
            
            # Define fields
            fields = [
                {"name": "title", "type": "text", "description": "The title of the skincare/dermatology blog post.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "author", "type": "text", "description": "The author or writer of the blog post.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "created_at", "type": "text", "description": "The creation timestamp.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "updated_at", "type": "text", "description": "The last update timestamp.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "tags", "type": "text[]", "description": "Array of tags denoting blog topics, categories, and keywords.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "summary", "type": "text", "description": "Brief summary of the blog post content.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "link", "type": "text", "description": "The URL link to the original blog article.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
                {"name": "content", "type": "text", "description": "The full text content of the article.", "range": None, "date_range": None, "groups": None, "date_median": None, "mean": None},
            ]
            
            # Define mappings
            mappings = {
                "document": {
                    "title": "title",
                    "author": "author",
                    "date": "created_at",
                    "content": "content",
                    "category": "summary",
                },
                "generic": {
                    "title": "title",
                    "subtitle": "summary",
                    "content": "content",
                    "url": "link",
                    "author": "author",
                    "timestamp": "created_at",
                    "tags": "tags",
                    "category": "summary",
                },
                "table": {
                    "title": "title",
                    "author": "author",
                    "created_at": "created_at",
                    "updated_at": "updated_at",
                    "tags": "tags",
                    "summary": "summary",
                    "link": "link",
                    "content": "content"
                }
            }
            
            blog_meta = {
                "name": blog_coll_name,
                "length": len_blog,
                "summary": "A collection containing 1,552 professional skincare and dermatology blogs. Each entry consists of an informative blog title, author name, creation/update timestamps, relevant topic tags, brief text summary, direct article URL link, and the full article content.",
                "index_properties": idx_props,
                "named_vectors": named_vecs,
                "vectorizer": vec,
                "fields": fields,
                "mappings": mappings
            }
            
            # Delete if exists in metadata
            if await preprocessed_collection_exists_async(blog_coll_name, cm):
                print(f"  Removing old preprocessed metadata for {blog_coll_name}...")
                await delete_preprocessed_collection_async(blog_coll_name, cm)
            
            # Write to metadata
            await metadata_coll.data.insert(blog_meta)
            print(f"  ✅ Preprocessed metadata for '{blog_coll_name}' written successfully!")

    print("\n" + "=" * 60)
    print("ALL DONE SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
