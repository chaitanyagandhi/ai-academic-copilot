from fastapi import FastAPI
from app.routes.ingest import router as ingest_router
from app.routes.search import router as search_router


app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
app.include_router(search_router, prefix="/search", tags=["search"])
