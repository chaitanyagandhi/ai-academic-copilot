from fastapi import FastAPI
from app.routes.ingest import router as ingest_router
from app.routes.search import router as search_router
from app.routes.chat import router as chat_router
from app.routes.ingest_pdf import router as ingest_pdf_router

from dotenv import load_dotenv



load_dotenv()

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
app.include_router(search_router, prefix="/search", tags=["search"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(ingest_pdf_router, prefix="/ingest", tags=["ingest"])


