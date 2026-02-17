# Use Python 3.12
FROM python:3.12

# UPDATE LINUX & INSTALL SYSTEM DEPENDENCIES
RUN apt-get update && apt-get install -y \
    libenchant-2-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
# Set working directory
WORKDIR /code

# Upgrades pip to the latest version
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the code
COPY . /code

# Create a writable directory for the database
RUN mkdir -p /code/data
RUN chmod 777 /code/data

# Run the app
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
