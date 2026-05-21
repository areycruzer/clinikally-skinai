#!/bin/bash
# CLINIKALLY SKINAI - VPS OVERNIGHT MASTER PROCESSOR
# Detached execution script that runs completely server-side

# Ensure we write everything to a single output log
exec > /app/master_run.log 2>&1
exec 2>&1

echo "=========================================================================="
echo "      CLINIKALLY AGENTIC SKINCARE AI - MASTER BACKGROUND RUNNER           "
echo "=========================================================================="
echo "Start Time : $(date)"
echo "Host       : $(hostname)"
echo "User       : $(whoami)"
echo "Directory  : $(pwd)"
echo "=========================================================================="

set -x # Enable verbose tracing of command execution

# 1. Stop active services to guarantee a pristine database state
echo "[STAGE 1] Stopping active containers..."
cd /app
docker compose down || true

# 2. Re-create and build services
echo "[STAGE 2] Building and raising Docker containers..."
docker compose up -d --build

# 3. Wait for Weaviate and services to warm up
echo "[STAGE 3] Waiting 20 seconds for Weaviate initialization..."
sleep 20

# 4. Sync datasets and ingestion scripts into the backend container
echo "[STAGE 4] Injecting files into backend container..."
docker cp /app/DermaGPT_Product_Database.xlsx app-backend-1:/DermaGPT_Product_Database.xlsx
docker cp /app/DermaGPT_Blogs_with_Metadata.zip app-backend-1:/DermaGPT_Blogs_with_Metadata.zip
docker cp /app/ingest_data.py app-backend-1:/app/ingest_data.py
docker cp /app/setup_collections.py app-backend-1:/app/setup_collections.py
docker cp /app/skinai/preprocessing/collection.py app-backend-1:/app/skinai/preprocessing/collection.py || true

# 5. Install standard spreadsheet dependencies inside container
echo "[STAGE 5] Installing runtime dependencies..."
docker compose exec -T backend pip install pandas openpyxl

# 6. Execute data ingestion
echo "[STAGE 6] Launching resilient data ingestion (2,999 products & ~1,552 blogs)..."
if docker compose exec -T backend python -u ingest_data.py; then
    echo "✅ [SUCCESS] Ingestion completed without fatal errors!"
else
    echo "❌ [FAILURE] Ingestion script crashed. Terminating."
    exit 1
fi

# 7. Execute Elysia/DSPy Preprocessing
echo "[STAGE 7] Starting DSPy preprocessing and collection schema setup..."
if docker compose exec -T backend python setup_collections.py; then
    echo "✅ [SUCCESS] Preprocessing setup completed successfully!"
else
    echo "❌ [FAILURE] Preprocessing script failed."
    exit 1
fi

# 8. Query Weaviate verification metrics
echo "[STAGE 8] Querying Weaviate DB metrics..."
curl -s http://127.0.0.1:8080/v1/nodes?output=verbose || true

echo "=========================================================================="
echo "✅ ALL STEPS EXECUTED SUCCESSFULLY!"
echo "End Time : $(date)"
echo "=========================================================================="
