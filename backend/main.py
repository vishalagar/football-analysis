
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import api
import os

app = FastAPI(title="AutoML Agent")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Router
app.include_router(api.router, prefix="/api")

# Mount Dataset for images
# Mount Dataset for images
import pathlib
BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "mlcc")
if os.path.exists(DATASET_DIR):
    app.mount("/dataset", StaticFiles(directory=DATASET_DIR), name="dataset")

# Static Files (Frontend)
FRONTEND_DIR = os.path.abspath("frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
