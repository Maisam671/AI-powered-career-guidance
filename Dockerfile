# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app folder into the container
COPY app ./app

# Copy the .env file if you want to use it locally (Render uses environment variables instead)
COPY .env .env

# Expose FastAPI port
EXPOSE 8000

# Set environment variables (Render overrides these automatically)
ENV PYTHONUNBUFFERED=1

# Command to start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


