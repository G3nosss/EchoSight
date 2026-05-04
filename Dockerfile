# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install Node.js
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency files first (for caching)
COPY package*.json ./
COPY requirements.txt ./

# Install Python and Node dependencies
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN npm install

# Copy the rest of the project
COPY . .

# Expose the port Node runs on
EXPOSE 3000

# Start the Node server (which spins up the Python daemons)
CMD ["node", "server.js"]
