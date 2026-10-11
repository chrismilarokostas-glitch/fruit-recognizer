import io
import re
import gc
import csv
import json
import base64
from datetime import timezone
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import matplotlib.cm as cm
from openpyxl import Workbook
from fastapi import FastAPI, UploadFile, File, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

import database
import models as db_models
from fruit_encyclopedia import get_fruit_info

db_models.Base.metadata.create_all(bind=database.engine)

with database.engine.connect() as conn:
    existing_columns = {row[1] for row in conn.execute(text("PRAGMA table_info(predictions)"))}
    if "gradcam_image" not in existing_columns:
        conn.execute(text("ALTER TABLE predictions ADD COLUMN gradcam_image TEXT"))
        conn.commit()
    if "client_id" not in existing_columns:
        conn.execute(text("ALTER TABLE predictions ADD COLUMN client_id VARCHAR"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_predictions_client_id ON predictions (client_id)"))
        conn.commit()

app = FastAPI(title="Fruit Recognition API")

# Ενεργοποίηση CORS για να μπορεί το Frontend (π.χ. React / HTML) να καλεί το API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
torch.set_num_threads(1)

MODEL_PATH = "fruit_mobilenetv2.pth"
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)

class_names = checkpoint['class_names']
num_classes = len(class_names)

model = models.mobilenet_v2()
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes)
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

IMAGE_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

GRADCAM_TARGET_LAYER = model.features[-1]

# Όριο μεγέθους upload - προστατεύει τη (λίγη) μνήμη του free tier από τεράστια αρχεία
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Το ιστορικό είναι προσωπικό ανά browser: το frontend παράγει ένα τυχαίο
# αναγνωριστικό (UUID, στο localStorage) και το στέλνει σε κάθε αίτημα. Κάθε
# εγγραφή "ανήκει" στο αναγνωριστικό που τη δημιούργησε, και μόνο αυτό μπορεί
# να τη δει, να την εξαγάγει ή να τη διαγράψει.
CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{16,64}$")


def utc_iso(dt):
    """
    Οι χρονοσημάνσεις αποθηκεύονται στη βάση σε UTC χωρίς ένδειξη ζώνης. Τις επιστρέφουμε
    με ρητό offset (+00:00), ώστε ο browser να τις μετατρέπει στην τοπική ώρα του χρήστη
    αντί να τις εκλαμβάνει ως τοπικές (που τις έδειχνε π.χ. 3 ώρες πίσω στην Ελλάδα).
    """
    return dt.replace(tzinfo=timezone.utc).isoformat() if dt else None


def get_client_id(x_client_id: str = Header(default="")) -> str:
    if not CLIENT_ID_PATTERN.match(x_client_id):
        raise HTTPException(status_code=400, detail="Λείπει ή δεν είναι έγκυρο το αναγνωριστικό X-Client-Id.")
    return x_client_id


def load_uploaded_image(image_bytes: bytes) -> Image.Image:
    """Ανοίγει τα bytes ως εικόνα RGB· αν δεν είναι έγκυρη εικόνα, επιστρέφει 400 αντί για 500."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()  # αναγκάζει πλήρη αποκωδικοποίηση τώρα (κομμένα/χαλασμένα αρχεία σκάνε εδώ)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Το αρχείο δεν είναι έγκυρη εικόνα ή είναι κατεστραμμένο.")

    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert('RGBA')
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert('RGB')


def generate_gradcam_data_uri(original_image: Image.Image, predicted_idx: int) -> str:
    """
    Παράγει Grad-CAM heatmap overlay για την προβλεπόμενη κλάση πάνω στην αρχική
    εικόνα, και το επιστρέφει ως base64 data URI για άμεση εμφάνιση στο frontend.
    """
    input_tensor = transform(original_image).unsqueeze(0)
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = GRADCAM_TARGET_LAYER.register_forward_hook(forward_hook)
    bh = GRADCAM_TARGET_LAYER.register_full_backward_hook(backward_hook)

    try:
        output = model(input_tensor)
        model.zero_grad()
        score = output[0, predicted_idx]
        score.backward()

        acts = activations[0][0]      # [C, H, W]
        grads = gradients[0][0]       # [C, H, W]
        weights = grads.mean(dim=(1, 2))

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for c in range(acts.shape[0]):
            cam += weights[c] * acts[c]

        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam_np = cam.detach().cpu().numpy()
    finally:
        fh.remove()
        bh.remove()
        model.zero_grad(set_to_none=True)
        gc.collect()

    cam_resized = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
        original_image.size, resample=Image.BILINEAR
    )
    cam_array = np.array(cam_resized) / 255.0

    colormap = cm.get_cmap("jet")
    heatmap_rgb = (colormap(cam_array)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb)

    blended = Image.blend(original_image.convert("RGB"), heatmap_img, alpha=0.45)

    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# --- ENDPOINTS ---
@app.get("/api/health")
def health_check():
    return {"message": "Fruit Recognition API is running successfully!"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db),
    client_id: str = Depends(get_client_id),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Το αρχείο πρέπει να είναι εικόνα.")

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Η εικόνα ξεπερνά το όριο των {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    image = load_uploaded_image(image_bytes)

    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

        # Top-3 πιθανότερες κλάσεις (για το "ΤΟΡ-3 ΠΙΘΑΝΟΤΗΤΕΣ" στο frontend)
        top3_confidences, top3_indices = torch.topk(probabilities, k=min(3, num_classes))

    predicted_fruit = class_names[predicted_idx.item()]
    confidence_score = round(confidence.item() * 100, 2)

    top3 = [
        {
            "fruit": class_names[idx.item()],
            "confidence": round(conf.item() * 100, 2),
        }
        for conf, idx in zip(top3_confidences, top3_indices)
    ]

    fruit_info = get_fruit_info(predicted_fruit)
    gradcam_image = generate_gradcam_data_uri(image, predicted_idx.item())

    db_record = db_models.PredictionRecord(
        filename=file.filename,
        fruit_name=predicted_fruit,
        confidence=confidence_score,
        gradcam_image=gradcam_image,
        client_id=client_id,
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return {
        "id": db_record.id,
        "filename": file.filename,
        "prediction": predicted_fruit,
        "confidence_percentage": confidence_score,
        "created_at": utc_iso(db_record.created_at),
        "top3": top3,
        "info": fruit_info,
        "gradcam_image": gradcam_image,
    }

@app.get("/history")
def get_history(db: Session = Depends(database.get_db), client_id: str = Depends(get_client_id)):
    records = (
        db.query(
            db_models.PredictionRecord.id,
            db_models.PredictionRecord.filename,
            db_models.PredictionRecord.fruit_name,
            db_models.PredictionRecord.confidence,
            db_models.PredictionRecord.created_at,
        )
        .filter(db_models.PredictionRecord.client_id == client_id)
        .order_by(db_models.PredictionRecord.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "fruit_name": r.fruit_name,
            "confidence": r.confidence,
            "created_at": utc_iso(r.created_at),
        }
        for r in records
    ]

def get_own_record(db: Session, record_id: int, client_id: str):
    """Η εγγραφή με αυτό το ID, μόνο αν ανήκει στον συγκεκριμένο browser (αλλιώς None -> 404)."""
    return (
        db.query(db_models.PredictionRecord)
        .filter(
            db_models.PredictionRecord.id == record_id,
            db_models.PredictionRecord.client_id == client_id,
        )
        .first()
    )


@app.get("/history/{record_id}/gradcam")
def get_history_gradcam(record_id: int, db: Session = Depends(database.get_db), client_id: str = Depends(get_client_id)):
    """Επιστρέφει το αποθηκευμένο Grad-CAM heatmap μιας συγκεκριμένης εγγραφής ιστορικού."""
    record = get_own_record(db, record_id, client_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Δεν βρέθηκε εγγραφή με ID {record_id}.")
    if not record.gradcam_image:
        raise HTTPException(status_code=404, detail="Δεν υπάρχει αποθηκευμένο Grad-CAM για αυτή την εγγραφή.")
    return {"id": record.id, "gradcam_image": record.gradcam_image}

@app.delete("/history/{record_id}")
def delete_history_record(record_id: int, db: Session = Depends(database.get_db), client_id: str = Depends(get_client_id)):
    """Διαγράφει μία μεμονωμένη εγγραφή ιστορικού με βάση το ID της."""
    record = get_own_record(db, record_id, client_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Δεν βρέθηκε εγγραφή με ID {record_id}.")
    db.delete(record)
    db.commit()
    return {"message": f"Η εγγραφή #{record_id} διαγράφηκε."}


class BulkDeleteRequest(BaseModel):
    ids: List[int]


@app.post("/history/bulk-delete")
def bulk_delete_history(payload: BulkDeleteRequest, db: Session = Depends(database.get_db), client_id: str = Depends(get_client_id)):
    """Διαγράφει πολλαπλές εγγραφές ιστορικού ταυτόχρονα (μαζική διαγραφή από το frontend)."""
    if not payload.ids:
        return {"deleted": 0}
    deleted = (
        db.query(db_models.PredictionRecord)
        .filter(
            db_models.PredictionRecord.id.in_(payload.ids),
            db_models.PredictionRecord.client_id == client_id,
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"deleted": deleted}

def get_own_records(db: Session, client_id: str):
    return (
        db.query(db_models.PredictionRecord)
        .filter(db_models.PredictionRecord.client_id == client_id)
        .order_by(db_models.PredictionRecord.created_at.desc())
        .all()
    )


@app.get("/export-excel")
def export_excel(db: Session = Depends(database.get_db), client_id: str = Depends(get_client_id)):
    """Εξαγωγή του ιστορικού προβλέψεων του χρήστη ως αρχείο Excel (.xlsx) για λήψη."""
    records = get_own_records(db, client_id)

    wb = Workbook()
    ws = wb.active
    ws.title = "Predictions"
    ws.append(["ID", "Filename", "Fruit Name", "Confidence (%)", "Created At (UTC)"])
    for r in records:
        ws.append([r.id, r.filename, r.fruit_name, r.confidence, r.created_at])

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=fruit_predictions_history.xlsx"},
    )

@app.get("/export-csv")
def export_csv(db: Session = Depends(database.get_db), client_id: str = Depends(get_client_id)):
    """Εξαγωγή του ιστορικού προβλέψεων του χρήστη ως αρχείο CSV για λήψη."""
    records = get_own_records(db, client_id)

    text_buffer = io.StringIO()
    writer = csv.writer(text_buffer)
    writer.writerow(["ID", "Filename", "Fruit Name", "Confidence (%)", "Created At (UTC)"])
    for r in records:
        writer.writerow([r.id, r.filename, r.fruit_name, r.confidence, r.created_at])

    output = io.BytesIO(text_buffer.getvalue().encode("utf-8-sig"))
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fruit_predictions_history.csv"},
    )

@app.get("/export-json")
def export_json(db: Session = Depends(database.get_db), client_id: str = Depends(get_client_id)):
    """Εξαγωγή του ιστορικού προβλέψεων του χρήστη ως αρχείο JSON για λήψη."""
    records = get_own_records(db, client_id)
    data = [
        {
            "id": r.id,
            "filename": r.filename,
            "fruit_name": r.fruit_name,
            "confidence": r.confidence,
            "created_at": utc_iso(r.created_at),
        }
        for r in records
    ]
    output = io.BytesIO(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
    return StreamingResponse(
        output,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=fruit_predictions_history.json"},
    )


# --- Frontend (static/) ---
app.mount("/", StaticFiles(directory="static", html=True), name="frontend")