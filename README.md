<img width="4388" height="3188" alt="image" src="https://github.com/user-attachments/assets/aa83730f-da05-48a4-87d7-327612111d6a" />

This is the high-level architecture for chat application with RAG integrating with OpenAI.

**1. Activate virtual environment in python**
     $ python3 -m venv <name for your venv>
     $ source <your_full_path_to>/gpt/bin/activate

**2. Install all required dependencies**
    $ pip install -r requirements.txt

**3. Run Docker containers for Otel-Collector, Milvus DB, Redis and Attu**
    A) Otel-Collector
    $ docker run -d --name otelcol \\n  -p 4317:4317 -p 4318:4318 \\n  -v "$PWD/collector.yaml":/etc/otelcol/config.yaml \\n  otel/opentelemetry-collector-contrib:latest \\n  --config /etc/otelcol/config.yaml
    B) Attu (Web UI for Milvus Vector DB)
    $ docker run --rm -d -p 8001:3000 --name attu zilliz/attu:latest\n
    C) Milvus
    $ docker compose -f docker/milvus-compose.yaml up -d
    D) Redis and Redis Insight (Same as Attu for Web UI access to Redis)
    $ docker compose -f docker/redis/docker-compose.yaml up -d

**4. Run Streamlit (frontend) and Main app**
    _opentelemetry-instrument _ is used to auto-instrument our Python App
    $ opentelemetry-instrument uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    $ opentelemetry-instrument streamlit run streamlit_app.py
