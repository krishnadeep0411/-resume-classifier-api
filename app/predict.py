import re

import torch
from app.model import load_model_and_vectorizer
from app.utils import extract_text_from_pdf

model, vectorizer, label_y, device = load_model_and_vectorizer()


def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9!?.,]", " ", text)
    return text.lower().strip()


def predict_resume_category(file):
    text = extract_text_from_pdf(file)
    cleaned_text = clean_text(text)

    vectorized = vectorizer.transform([cleaned_text]).toarray()
    input_tensor = torch.tensor(vectorized, dtype=torch.float).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, dim=1)

    return label_y[pred.item()]
