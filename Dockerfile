# Python lightweight container for AI service
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --retries 10 --timeout 100 Flask Pillow google-genai python-dotenv requests gunicorn
RUN pip install --no-cache-dir --retries 10 --timeout 100 numpy scipy scikit-learn
RUN pip install --no-cache-dir --retries 10 --timeout 100 cffi cryptography pydantic pydantic_core typing_extensions
RUN pip install --no-cache-dir --retries 10 --timeout 100 langchain-core langsmith langgraph-checkpoint langgraph-prebuilt langgraph
COPY . .
EXPOSE 8000
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
