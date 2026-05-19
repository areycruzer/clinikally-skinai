#!/usr/bin/env python3
"""Setup script with aggressive timeout/retry for free OpenRouter models."""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configure litellm BEFORE any imports
os.environ["LITELLM_NUM_RETRIES"] = "10"
os.environ["LITELLM_REQUEST_TIMEOUT"] = "120"
os.environ["LITELLM_RETRY_DELAY"] = "15"

import warnings
warnings.filterwarnings("ignore")

import litellm
litellm.request_timeout = 120
litellm.num_retries = 10
litellm.retry_after = 15
litellm.set_verbose = False

# Quick test: can we actually call the model?
from dotenv import load_dotenv
load_dotenv(override=True)

print("=" * 60)
print("STEP 1: Testing LLM connectivity...")
print("=" * 60)

api_key = os.getenv("OPENROUTER_API_KEY")
base_model = os.getenv("BASE_MODEL")
complex_model = os.getenv("COMPLEX_MODEL")
print(f"  BASE_MODEL: {base_model}")
print(f"  COMPLEX_MODEL: {complex_model}")

# Test base model directly via litellm
try:
    response = litellm.completion(
        model=f"openrouter/{base_model}",
        messages=[{"role": "user", "content": "Say hello in 3 words."}],
        api_key=api_key,
        max_tokens=50,
        timeout=60,
        num_retries=5,
    )
    print(f"  ✅ Base model works: {response.choices[0].message.content.strip()}")
except Exception as e:
    print(f"  ❌ Base model failed: {e}")
    print("  Trying alternative model...")
    # Try alternatives in order
    alternatives = [
        "qwen/qwen3-coder:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "nvidia/nemotron-nano-9b-v2:free",
        "openai/gpt-oss-20b:free",
    ]
    for alt in alternatives:
        try:
            response = litellm.completion(
                model=f"openrouter/{alt}",
                messages=[{"role": "user", "content": "Say hello in 3 words."}],
                api_key=api_key,
                max_tokens=50,
                timeout=60,
            )
            print(f"  ✅ Alternative '{alt}' works: {response.choices[0].message.content.strip()}")
            base_model = alt
            os.environ["BASE_MODEL"] = alt
            break
        except Exception as e2:
            print(f"  ❌ {alt}: {e2}")
    else:
        print("  ❌ No free models available. Exiting.")
        sys.exit(1)

# Also test complex model
print()
try:
    response = litellm.completion(
        model=f"openrouter/{complex_model}",
        messages=[{"role": "user", "content": "Say hello in 3 words."}],
        api_key=api_key,
        max_tokens=50,
        timeout=60,
        num_retries=5,
    )
    print(f"  ✅ Complex model works: {response.choices[0].message.content.strip()}")
except Exception as e:
    print(f"  ❌ Complex model failed: {e}")
    # Use same as base
    complex_model = base_model
    os.environ["COMPLEX_MODEL"] = base_model
    print(f"  ↪ Falling back to: {complex_model}")

print()
print(f"  Final BASE_MODEL: {base_model}")
print(f"  Final COMPLEX_MODEL: {complex_model}")

# Now import skinai and run preprocessing
from skinai.config import settings, configure
settings.set_from_env()
settings.BASE_MODEL = base_model
settings.COMPLEX_MODEL = complex_model

from skinai import preprocess, preprocessed_collection_exists

MAX_RETRIES = 5
RETRY_DELAY = 30

for collection_name in ["SkincareProducts", "SkincareBlogs"]:
    print()
    print("=" * 60)
    print(f"STEP: Preprocessing '{collection_name}'...")
    print("=" * 60)
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            preprocess(
                collection_name,
                force=True,
                min_sample_size=5,
                max_sample_size=10,
                num_sample_tokens=5000
            )
            print(f"  ✅ {collection_name} preprocessing complete!")
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RateLimit" in err_str or "rate" in err_str.lower():
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * attempt
                    print(f"  ⏳ Rate limited (attempt {attempt}/{MAX_RETRIES}). Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  ❌ Failed after {MAX_RETRIES} attempts: {err_str[:200]}")
            else:
                print(f"  ❌ Error: {err_str[:300]}")
                break
    
    time.sleep(15)

# Verify
print()
print("=" * 60)
print("VERIFICATION")
print("=" * 60)
products_ok = preprocessed_collection_exists("SkincareProducts")
blogs_ok = preprocessed_collection_exists("SkincareBlogs")
print(f"  SkincareProducts: {'✅' if products_ok else '❌'}")
print(f"  SkincareBlogs: {'✅' if blogs_ok else '❌'}")

if products_ok and blogs_ok:
    print("\n🎉 Both collections preprocessed and ready!")
else:
    print("\n⚠️  Some collections failed.")
    sys.exit(1)
