from fastapi import FastAPI

app = FastAPI(
    title="ML Stack Lab API",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"message": "ML Stack Lab API"}


@app.get("/health")
def health():
    return {"status": "ok"}
