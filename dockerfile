FROM python:3.11-slim

# ENV PATH handles venv activation for ALL subsequent RUN steps and container runtime
# No need for "source .venv/bin/activate" at all
RUN python3 -m venv /.venv
ENV PATH="/.venv/bin:$PATH"

# Packages available to the agent-generated code inside the container
RUN pip install numpy pandas scipy sympy matplotlib seaborn scikit-learn

# Create the temp directory where solution.py files will live
RUN mkdir -p /app/tmp

WORKDIR /app/tmp