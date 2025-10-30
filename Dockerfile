# 1️⃣ Use official Python image
FROM python:3.11-slim

# 2️⃣ Environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# 3️⃣ Install OS dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# 4️⃣ Set working directory
WORKDIR /app

# 5️⃣ Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt


# 6️⃣ Copy the rest of the project files
# COPY app.py .
# COPY best.pt .
# COPY templates/ templates/
# COPY static/ static/

COPY . .
# 7️⃣ Expose Flask port
EXPOSE 5000

# 8️⃣ Run Flask app
# CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "1", "app:app"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

