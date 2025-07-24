# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Dash default port
EXPOSE 8050

# Command to run the Dash app
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--workers", "1"]
