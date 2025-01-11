from fastapi import FastAPI
from src.eduai_detector.interface.api import app

# Create a new FastAPI app for Vercel
vercel_app = FastAPI()

# Include the routes from the main app
vercel_app.include_router(app.router)

# Export for Vercel
app = vercel_app 