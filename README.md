# Semi-supervised Ticket Triage (scikit-learn best-practice style)

This mini project demonstrates a **real-world scenario**: support ticket classification when you have:
- a small labeled dataset (expensive manual labels),
- and a larger unlabeled dataset (historical tickets).

It follows a practical approach aligned with scikit-learn unsupervised-learning usage:
1. Learn structure from unlabeled text using clustering.
2. Convert high-confidence clusters into pseudo-labels.
3. Train a supervised classifier on labeled + pseudo-labeled data.
4. Evaluate only on a clean held-out labeled test set.

---

## Why this is useful

In operations teams, manual labeling is slow.  
This approach helps bootstrap a production classifier while reducing annotation effort.

---

## Source tree

- `src/python/classification_from_unsupervised.py`  
  Core training logic + prediction helper.
- `src/python/api.py`  
  FastAPI wrapper exposing `/predict` and `/metrics`.
- `src/frontend/`  
  Minimal React UI (Vite) for testing predictions.

---

## Best-practice notes (important)

- **No test leakage**: hold-out labeled test samples are not used for pseudo-label generation.
- **Pipeline thinking**: consistent vectorization + model training workflow.
- **Confidence filtering**: keep only close-to-centroid pseudo-labels.
- **Balanced classifier**: `class_weight="balanced"` to reduce class imbalance impact.
- **Dual evaluation**:
  - intrinsic unsupervised quality (optionally silhouette, not shown),
  - and downstream classification quality (macro F1 shown).

For production, add:
- stronger validation (cross-validation, temporal split),
- human review loop for low-confidence predictions,
- drift monitoring and periodic re-training.

---

## Run backend (Python)

```bash
cd src/python
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Test quickly:
- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/metrics`
- `POST http://127.0.0.1:8000/predict` with JSON:
  ```json
  {"text":"password reset does not work"}
  ```

---

## Run frontend (Node + React)

```bash
cd src/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

---

## Example use case flow

1. Agent pastes a new support message in UI.
2. Frontend calls `/predict`.
3. API returns predicted queue label (e.g., `billing`, `auth`, `bug`) and confidence.
4. High-confidence items auto-route; low-confidence items go to human triage.

---

## Tech stack

- Python 3.10+
- scikit-learn
- FastAPI
- Node.js + React (Vite)

---

## Limitations (intentional for lightweight demo)

- Tiny toy dataset.
- Simple pseudo-labeling strategy.
- No persistence/database/auth.

Still, the architecture is a solid starting template for a real implementation.