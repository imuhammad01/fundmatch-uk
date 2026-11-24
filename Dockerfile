FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY app.py ./app.py
COPY artifacts ./artifacts
COPY data ./data
COPY .streamlit ./.streamlit
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

EXPOSE 8765
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8765", "--server.address", "0.0.0.0", "--server.headless", "true"]
