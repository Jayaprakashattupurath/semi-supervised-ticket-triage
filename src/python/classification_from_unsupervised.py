from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class TicketModel:
    vectorizer: TfidfVectorizer
    clusterer: MiniBatchKMeans
    classifier: LogisticRegression
    cluster_to_label: Dict[int, str]
    confidence_threshold: float


def build_demo_data() -> Tuple[List[str], List[str], List[str]]:
    # Small labeled set (expensive, manually reviewed)
    labeled_texts = [
        "I was charged twice this month",
        "refund not received for my order",
        "invoice amount is incorrect",
        "cannot login to my account",
        "password reset link expired",
        "two factor code not working",
        "app crashes when uploading photo",
        "screen freezes after update",
        "error 500 when saving profile",
    ]
    labeled_y = [
        "billing", "billing", "billing",
        "auth", "auth", "auth",
        "bug", "bug", "bug",
    ]

    # Larger unlabeled pool (real-world: historical tickets)
    unlabeled_texts = [
        "payment failed but money deducted",
        "subscription cancellation refund status",
        "credit card charged after cancellation",
        "login page keeps reloading",
        "cannot access dashboard after password change",
        "otp verification fails every time",
        "mobile app closes instantly",
        "unexpected server error while submitting form",
        "profile picture upload causes crash",
        "why was i billed for annual plan",
        "locked out of account after update",
        "application hangs on checkout step",
    ]
    return labeled_texts, labeled_y, unlabeled_texts


def train_and_build_predictor(random_state: int = 42) -> Tuple[TicketModel, Dict]:
    labeled_texts, labeled_y, unlabeled_texts = build_demo_data()

    # Best practice: keep a clean supervised hold-out for honest evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        labeled_texts, labeled_y, test_size=0.33, stratify=labeled_y, random_state=random_state
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_unlab_vec = vectorizer.transform(unlabeled_texts)
    X_test_vec = vectorizer.transform(X_test)

    # Unsupervised structure on train+unlabeled (not test)
    X_cluster_pool = np.vstack([X_train_vec.toarray(), X_unlab_vec.toarray()])
    clusterer = MiniBatchKMeans(n_clusters=3, random_state=random_state, batch_size=16)
    cluster_ids_pool = clusterer.fit_predict(X_cluster_pool)

    train_cluster_ids = cluster_ids_pool[: len(X_train)]
    unlab_cluster_ids = cluster_ids_pool[len(X_train):]

    # Map clusters -> known class using labeled train majority vote
    cluster_to_label = {}
    for c in np.unique(train_cluster_ids):
        idx = np.where(train_cluster_ids == c)[0]
        labels = [y_train[i] for i in idx]
        majority = max(set(labels), key=labels.count)
        cluster_to_label[int(c)] = majority

    # Pseudo-label unlabeled points using cluster mapping
    pseudo_X = []
    pseudo_y = []

    # Confidence from distance to assigned centroid (lower distance = higher confidence)
    distances = clusterer.transform(X_unlab_vec.toarray())
    assigned_dist = distances[np.arange(len(unlabeled_texts)), unlab_cluster_ids]
    threshold = float(np.quantile(assigned_dist, 0.70))  # keep top ~70% confidence

    for i, c in enumerate(unlab_cluster_ids):
        if assigned_dist[i] <= threshold and int(c) in cluster_to_label:
            pseudo_X.append(unlabeled_texts[i])
            pseudo_y.append(cluster_to_label[int(c)])

    X_supervised = X_train + pseudo_X
    y_supervised = y_train + pseudo_y

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    clf.fit(vectorizer.transform(X_supervised), y_supervised)

    y_pred = clf.predict(X_test_vec)
    metrics = {
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "train_labeled": len(X_train),
        "train_pseudo_labeled": len(pseudo_X),
        "test_size": len(X_test),
    }

    model = TicketModel(
        vectorizer=vectorizer,
        clusterer=clusterer,
        classifier=clf,
        cluster_to_label=cluster_to_label,
        confidence_threshold=threshold,
    )
    return model, metrics


def predict_ticket(model: TicketModel, text: str) -> Dict:
    vec = model.vectorizer.transform([text])
    probs = model.classifier.predict_proba(vec)[0]
    labels = model.classifier.classes_
    top_idx = int(np.argmax(probs))
    return {
        "predicted_label": str(labels[top_idx]),
        "confidence": float(probs[top_idx]),
        "all_scores": {str(labels[i]): float(probs[i]) for i in range(len(labels))},
    }


if __name__ == "__main__":
    model, metrics = train_and_build_predictor()
    print("Quick evaluation (hold-out labeled set):")
    print(f"Macro F1: {metrics['f1_macro']:.3f}")
    print(f"Pseudo-labeled samples used: {metrics['train_pseudo_labeled']}")
    example = "I cannot sign in and reset email is not arriving"
    print("Example:", predict_ticket(model, example))