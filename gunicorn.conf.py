# gunicorn.conf.py
# Extended timeout for AI LangGraph multi-node execution
timeout = 300
workers = 2
threads = 4
keepalive = 5
bind = "0.0.0.0:8000"
