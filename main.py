# main.py
from src.eduai_detector.interface.api import app

# For Vercel deployment, we need to expose the app directly
app = app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)