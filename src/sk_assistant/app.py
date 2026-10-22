from fastapi import FastAPI


app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check: returns ok if the server is running"""

    return {"status": "ok"}