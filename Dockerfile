# Use a base Python image compatible with your requirements.txt
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code to the container
# This will copy app_dash.py, assets/, CSVs etc.
COPY . .

# Expose the port your Dash app will run on (default 8050)
EXPOSE 8050

# Command to run your Dash app using Gunicorn
# This should match how you run it locally with gunicorn
CMD ["gunicorn", "app_dash:server", "--bind", "0.0.0.0:8050"]