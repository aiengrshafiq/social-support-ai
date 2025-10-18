# docker/Dockerfile.ml

# Use the same base image as the main API for consistency
FROM python:3.11-slim

WORKDIR /app

# Install specific dependencies for the ML service
# Keep this minimal!
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    scikit-learn \
    pandas \
    joblib

# Copy ONLY the 'ml' directory into the container
COPY ml/ /app/ml/

# Expose the port the ML service will run on (different from main API)
EXPOSE 8001

# Command to run the ML service
CMD ["uvicorn", "ml.serve_model:app", "--host", "0.0.0.0", "--port", "8001"]