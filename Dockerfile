# Use a lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app .

# Expose FastAPI port
EXPOSE 8000

# Set environment variables
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV WEAVIATE_CLOUD_URL=${WEAVIATE_CLOUD_URL}
ENV WEAVIATE_API_KEY=${WEAVIATE_API_KEY}

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
