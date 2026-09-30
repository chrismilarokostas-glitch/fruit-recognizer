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
# ΣΗΜΑΝΤΙΚΟ: το --index-url περιορίζει το pip ΜΟΝΟ σε αυτό το index, το οποίο
# δεν έχει wheel για transitive dependencies όπως το typing_extensions -
# χωρίς wheel, το pip προσπαθεί να το χτίσει από πηγαίο κώδικα και σκάει
# (χρειάζεται flit_core που επίσης δεν υπάρχει εκεί). Το εγκαθιστούμε πρώτα
# από το κανονικό PyPI ώστε να είναι ήδη ικανοποιημένο πριν περιοριστούμε.
RUN pip install --no-cache-dir typing_extensions \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py database.py models.py fruit_encyclopedia.py fruit_mobilenetv2.pth ./
COPY static/ ./static/

# Το Render (και οι περισσότερες PaaS) περνάνε το πραγματικό port μέσω $PORT -
# το 8000 είναι απλά το fallback για τοπικό `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
