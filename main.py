import io
import csv
import json
import base64
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.cm as cm
from openpyxl import Workbook
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

import database
import models as db_models
from fruit_encyclopedia import get_fruit_info

# 1. Αρχικοποίηση Βάσης
db_models.Base.metadata.create_all(bind=database.engine)

# Ελαφριά migration: το `create_all` δεν προσθέτει νέες στήλες σε ήδη υπάρχοντα
# πίνακα, οπότε ελέγχουμε χειροκίνητα αν λείπει η gradcam_image (π.χ. σε βάση
# που δημιουργήθηκε πριν προστεθεί αυτό το feature) και την προσθέτουμε.
with database.engine.connect() as conn:
    existing_columns = {row[1] for row in conn.execute(text("PRAGMA table_info(predictions)"))}
    if "gradcam_image" not in existing_columns:
        conn.execute(text("ALTER TABLE predictions ADD COLUMN gradcam_image TEXT"))
        conn.commit()

# 2. Αρχικοποίηση FastAPI
app = FastAPI(title="Fruit Recognition API")

# Ενεργοποίηση CORS για να μπορεί το Frontend (π.χ. React / HTML) να καλεί το API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Φόρτωση Μοντέλου PyTorch
MODEL_PATH = "fruit_mobilenetv2.pth"
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)

class_names = checkpoint['class_names']
num_classes = len(class_names)

# Δημιουργία αρχιτεκτονικής MobileNetV2
model = models.mobilenet_v2()
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes)
)

# Φόρτωση των εκπαιδευμένων βαρών
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 4. Image Preprocessing Transform
IMAGE_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Το τελευταίο conv block του MobileNetV2 - εκεί "ζουν" τα πιο υψηλού επιπέδου
# χωρικά χαρακτηριστικά (σχήμα/υφή αντικειμένου), ίδιο layer με το grad_cam.py
GRADCAM_TARGET_LAYER = model.features[-1]


def generate_gradcam_data_uri(original_image: Image.Image, predicted_idx: int) -> str:
    """
    Παράγει Grad-CAM heatmap overlay για την προβλεπόμενη κλάση πάνω στην αρχική
    εικόνα, και το επιστρέφει ως base64 data URI (έτοιμο να μπει απευθείας σε
    ένα <img src="..."> στο frontend, χωρίς ενδιάμεσο αρχείο).
    """
    input_tensor = transform(original_image).unsqueeze(0)
    input_tensor.requires_grad_(True)

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
    # Μετακινήθηκε από το "/" στο "/api/health" ώστε το "/" να είναι ελεύθερο
    # για να σερβίρει το frontend (static/index.html) - βλ. το mount στο τέλος
    # του αρχείου.
    return {"message": "Fruit Recognition API is running successfully!"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    save_to_history: bool = Query(default=True, description="False για live camera frames - δεν αποθηκεύεται στη βάση"),
    db: Session = Depends(database.get_db),
):
    # Έλεγχος αν είναι εικόνα
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Το αρχείο πρέπει να είναι εικόνα.")

# Ανάγνωση εικόνας
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Μετατροπή RGBA σε RGB με ΛΕΥΚΟ φόντο (αν είναι PNG)
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    else:
        image = image.convert('RGB')
    
    # Προεπεξεργασία & Πρόβλεψη
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

    # Grad-CAM: ΜΟΝΟ για κανονικά uploads, όχι για live camera frames - το backward
    # pass προσθέτει καθυστέρηση που θα χαλούσε την εμπειρία του live streaming
    # (νέο frame κάθε 800ms). Υπολογίζεται πριν την αποθήκευση ώστε να μπει
    # στην ίδια εγγραφή του ιστορικού, όχι μόνο στην άμεση απάντηση.
    gradcam_image = None
    if save_to_history:
        gradcam_image = generate_gradcam_data_uri(image, predicted_idx.item())

    # Αποθήκευση στη Βάση Δεδομένων - ΜΟΝΟ αν δεν είναι live camera frame.
    # (Το live camera στέλνει save_to_history=false, ώστε να μη "πλημμυρίζει"
    # τη βάση με μία εγγραφή κάθε 800ms όσο είναι ενεργό το stream.)
    db_record = None
    if save_to_history:
        db_record = db_models.PredictionRecord(
            filename=file.filename,
            fruit_name=predicted_fruit,
            confidence=confidence_score,
            gradcam_image=gradcam_image
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

    return {
        "id": db_record.id if db_record else None,
        "filename": file.filename,
        "prediction": predicted_fruit,
        "confidence_percentage": confidence_score,
        "created_at": db_record.created_at if db_record else None,
        "top3": top3,
        "info": fruit_info,
        "gradcam_image": gradcam_image,
    }

@app.get("/history")
def get_history(db: Session = Depends(database.get_db)):
    # Εσκεμμένα ΔΕΝ επιστρέφουμε τη στήλη gradcam_image εδώ - είναι base64 εικόνα
    # ανά εγγραφή και θα έκανε τη λίστα ιστορικού πολύ βαριά να φορτώσει.
    # Το Grad-CAM κάθε εγγραφής ανακτάται ξεχωριστά, on-demand, από το
    # /history/{record_id}/gradcam όταν ο χρήστης θέλει να το δει.
    records = (
        db.query(
            db_models.PredictionRecord.id,
            db_models.PredictionRecord.filename,
            db_models.PredictionRecord.fruit_name,
            db_models.PredictionRecord.confidence,
            db_models.PredictionRecord.created_at,
        )
        .order_by(db_models.PredictionRecord.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "fruit_name": r.fruit_name,
            "confidence": r.confidence,
            "created_at": r.created_at,
        }
        for r in records
    ]

@app.get("/history/{record_id}/gradcam")
def get_history_gradcam(record_id: int, db: Session = Depends(database.get_db)):
    """Επιστρέφει το αποθηκευμένο Grad-CAM heatmap μιας συγκεκριμένης εγγραφής ιστορικού."""
    record = db.query(db_models.PredictionRecord).filter(db_models.PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Δεν βρέθηκε εγγραφή με ID {record_id}.")
    if not record.gradcam_image:
        raise HTTPException(status_code=404, detail="Δεν υπάρχει αποθηκευμένο Grad-CAM για αυτή την εγγραφή.")
    return {"id": record.id, "gradcam_image": record.gradcam_image}

@app.delete("/history/{record_id}")
def delete_history_record(record_id: int, db: Session = Depends(database.get_db)):
    """Διαγράφει μία μεμονωμένη εγγραφή ιστορικού με βάση το ID της."""
    record = db.query(db_models.PredictionRecord).filter(db_models.PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Δεν βρέθηκε εγγραφή με ID {record_id}.")
    db.delete(record)
    db.commit()
    return {"message": f"Η εγγραφή #{record_id} διαγράφηκε."}


class BulkDeleteRequest(BaseModel):
    ids: List[int]


@app.post("/history/bulk-delete")
def bulk_delete_history(payload: BulkDeleteRequest, db: Session = Depends(database.get_db)):
    """Διαγράφει πολλαπλές εγγραφές ιστορικού ταυτόχρονα (μαζική διαγραφή από το frontend)."""
    if not payload.ids:
        return {"deleted": 0}
    deleted = (
        db.query(db_models.PredictionRecord)
        .filter(db_models.PredictionRecord.id.in_(payload.ids))
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"deleted": deleted}

@app.get("/export-excel")
def export_excel(db: Session = Depends(database.get_db)):
    """Εξαγωγή όλου του ιστορικού προβλέψεων ως αρχείο Excel (.xlsx) για λήψη."""
    records = db.query(db_models.PredictionRecord).order_by(db_models.PredictionRecord.created_at.desc()).all()

    wb = Workbook()
    ws = wb.active
    ws.title = "Predictions"
    ws.append(["ID", "Filename", "Fruit Name", "Confidence (%)", "Created At"])
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
def export_csv(db: Session = Depends(database.get_db)):
    """Εξαγωγή όλου του ιστορικού προβλέψεων ως αρχείο CSV για λήψη."""
    records = db.query(db_models.PredictionRecord).order_by(db_models.PredictionRecord.created_at.desc()).all()

    text_buffer = io.StringIO()
    writer = csv.writer(text_buffer)
    writer.writerow(["ID", "Filename", "Fruit Name", "Confidence (%)", "Created At"])
    for r in records:
        writer.writerow([r.id, r.filename, r.fruit_name, r.confidence, r.created_at])

    # utf-8-sig (BOM) ώστε τα ελληνικά να ανοίγουν σωστά αν το CSV ανοιχτεί στο Excel
    output = io.BytesIO(text_buffer.getvalue().encode("utf-8-sig"))
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fruit_predictions_history.csv"},
    )

@app.get("/export-json")
def export_json(db: Session = Depends(database.get_db)):
    """Εξαγωγή όλου του ιστορικού προβλέψεων ως αρχείο JSON για λήψη."""
    records = db.query(db_models.PredictionRecord).order_by(db_models.PredictionRecord.created_at.desc()).all()
    data = [
        {
            "id": r.id,
            "filename": r.filename,
            "fruit_name": r.fruit_name,
            "confidence": r.confidence,
            "created_at": r.created_at.isoformat() if r.created_at else None,
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
# ΠΡΕΠΕΙ να μπει ΤΕΛΕΥΤΑΙΟ: το Starlette router ελέγχει τα routes με τη σειρά
# καταχώρησής τους, οπότε όλα τα παραπάνω API endpoints (πιο συγκεκριμένα paths)
# "πιάνουν" πρώτα το ταίριασμα, και μόνο ό,τι δεν ταιριάζει με API route (π.χ.
# "/", "/styles.css", "/app.js") καταλήγει εδώ, στα static αρχεία.
app.mount("/", StaticFiles(directory="static", html=True), name="frontend")