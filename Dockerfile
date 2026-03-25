FROM python:3.12

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

# UPDATE LINUX & INSTALL SYSTEM DEPENDENCIES
RUN apt-get update && apt-get install -y \
    libenchant-2-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
# Set working directory
WORKDIR /code

# Copy uv project files and install dependencies
COPY ./pyproject.toml /code/pyproject.toml
COPY ./uv.lock /code/uv.lock
RUN uv sync --frozen --no-dev

ENV PATH="/code/.venv/bin:${PATH}"
ENV HF_HOME="/code/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/code/.cache/huggingface/hub"
ENV SENTENCE_TRANSFORMERS_HOME="/code/.cache/sentence-transformers"

# Warm the Hugging Face caches during build so container startup does not block
# on model downloads in constrained runtimes such as Spaces cold starts.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L12-v2')" \
    && python -c "from transformers import DistilBertTokenizer, DistilBertModel; DistilBertTokenizer.from_pretrained('distilbert-base-uncased'); DistilBertModel.from_pretrained('distilbert-base-uncased')"

# Copy the rest of the code
COPY . /code

# Create a writable directory for the database
RUN mkdir -p /code/data
RUN chmod 777 /code/data

# Run the app
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
