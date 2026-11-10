from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from classification_from_unsupervised import train_and_build_predictor, predict_ticket

app = FastAPI(title="Semi-supervised Ticket Triage API")
model, metrics = train_and_build_predictor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TicketIn(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    return metrics

@app.post("/predict")
def predict(data: TicketIn):
    return predict_ticket(model, data.text)