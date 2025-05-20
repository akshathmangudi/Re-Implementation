FROM python:3.11-slim

# Set working directory
WORKDIR /refrakt

# Copy all code into container
COPY . .

# Install your library (assumes setup.py or pyproject.toml exists)
RUN pip install --no-cache-dir .

# Optional: open a Python shell by default when running container
CMD ["python3"]
