FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install OS dependencies (if needed)
RUN apt-get update && apt-get install -y gcc

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy API code
COPY . .

# Expose port 8000
EXPOSE 8000

# Start API (reload for dev, remove for prod)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
