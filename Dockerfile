FROM python:3.10-slim

WORKDIR /app

# matplotlib.cm δεν χρειάζεται GUI backend (δεν χρησιμοποιούμε pyplot), αλλά
# το ορίζουμε ρητά ώστε να μην προσπαθήσει ποτέ να φορτώσει κάποιο interactive
# backend μέσα στο container.
ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

# Το CPU-only PyTorch index είναι πολύ μικρότερο από το default PyPI index
# (χωρίς CUDA binaries) - χρειαζόμαστε μόνο CPU inference σε production.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py database.py models.py fruit_encyclopedia.py fruit_mobilenetv2.pth ./
COPY static/ ./static/

# Το Render (και οι περισσότερες PaaS) περνάνε το πραγματικό port μέσω $PORT -
# το 8000 είναι απλά το fallback για τοπικό `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
