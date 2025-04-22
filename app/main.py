from fastapi import FastAPI, UploadFile, File
from app.predict import predict_resume_category

app = FastAPI(title="Resume Classifier API", description="Upload a resume PDF to classify the job domain.",
              version="1.0")


@app.post("/predict", summary="Predict Resume Category")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    category = predict_resume_category(file)
    return {"predicted_category": category}

