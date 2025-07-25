from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI! Test Deployment 2 ! "}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
