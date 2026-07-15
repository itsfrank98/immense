import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from kornia.losses import binary_focal_loss_with_logits
from modelling.word_embedding_bert import WordEmb

# --------------------------------------------------
# Classe SoftmaxClassifier
# --------------------------------------------------
class SoftmaxClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, loss="weighted", weights=None):
        """
        Classificatore Softmax dinamico.
        """
        super(SoftmaxClassifier, self).__init__()
        self.device = torch.device("cpu")
        self.loss = loss
        self.weights = weights.to(self.device) if weights is not None else None

        self.fc1 = nn.Linear(embedding_dim, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.dropout = nn.Dropout(0.3)

        self.to("cpu")

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to("cpu", dtype=torch.float32)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)

        if self.loss == "weighted":
            out = torch.log_softmax(out, dim=-1)
        return out

    def train_model(self, train_loader, val_loader=None, optimizer=None, num_epochs=50, learning_rate=0.001):
        device = self.device
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if self.weights is not None:
            self.weights = self.weights.to(device)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.train()
            total_loss, train_correct, train_total = 0, 0, 0

            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                out = self(batch_x)

                if self.loss == "focal":
                    target = torch.tensor(np.eye(self.fc2.out_features, dtype='uint8')[batch_y.cpu()],
                                          dtype=torch.float).to(self.device)
                    loss = binary_focal_loss_with_logits(out, target=target, reduction="mean")
                elif self.loss == "weighted":
                    loss = nn.functional.nll_loss(out, batch_y.long(), weight=self.weights)
                else:
                    loss = nn.functional.cross_entropy(out, batch_y.long())

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(out.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)

            # Validation
            if val_loader is not None:
                val_loss, val_accuracy, _ = self.test(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_accuracy)
                if val_loss < best_loss:
                    best_loss = val_loss
                    print(f"New best model found at epoch {epoch + 1}. Val Loss: {val_loss:.4f}")

            if epoch % 5 == 0:
                print(f"\nEpoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
                if val_loader is not None:
                    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        return history

    @torch.no_grad()
    def test(self, data_loader):
        """
        Valutazione su un DataLoader.
        """
        self.eval()
        device = self.device
        if self.weights is not None:
            self.weights = self.weights.to(device)
        total_loss, correct, total = 0, 0, 0
        all_preds = []

        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            out = self(batch_x)

            if self.loss == "focal":
                target = torch.tensor(
                    np.eye(self.fc2.out_features, dtype='uint8')[batch_y.cpu()],
                    dtype=torch.float
                ).to(self.device)
                loss = binary_focal_loss_with_logits(out, target=target, reduction="mean")
            elif self.loss == "weighted":
                loss = nn.functional.nll_loss(out, batch_y.long(), weight=self.weights)
            else:
                loss = nn.functional.cross_entropy(out, batch_y.long())

            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            all_preds.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy, np.array(all_preds)

    @torch.no_grad()
    def predict_proba(self, X):
        self.eval()

        # Conversione sicura a tensore float
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        else:
            X = X.float()

        X = X.to("cpu")

        logits = self(X)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        return probs.detach().cpu().numpy()

    @torch.no_grad()
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)


# --------------------------------------------------
# MAIN: train o test
# --------------------------------------------------
def train_softmax_classifier(mode="train"):
    if mode == "train":
        dataset_path = "dataset/train.csv"
        emb_file = "models/embeddings_train_content_relation_spatial.pt"
        lbl_file = "models/labels_train_content_relation_spatial.pt"
    else:
        dataset_path = "dataset/test.csv"
        emb_file = "models/embeddings_test_content_relation_spatial.pt"
        lbl_file = "models/labels_test_content_relation_spatial.pt"

    print(f"[SoftmaxClassifier] Mode: {mode}, usando file: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"[SoftmaxClassifier] Dataset caricato con shape: {df.shape}")

    # --- Embeddings & Labels ---
    embeddings, labels, _, _ = get_or_create_bert_embeddings(
        emb_file=emb_file,
        lbl_file=lbl_file,
        dataset_path=dataset_path,
        id_field_name="id",
        text_field_name="text_cleaned",
        label_field_name="label"
    )

    device = torch.device("cpu")

    # --- Training ---
    if mode == "train":
        X_t = torch.tensor(embeddings, dtype=torch.float)
        y_t = torch.tensor(labels, dtype=torch.long)
        train_ds = TensorDataset(X_t, y_t)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        classifier = SoftmaxClassifier(
            embedding_dim=X_t.shape[1],
            num_classes=len(np.unique(labels))
        ).to(device)

        classifier.train_model(train_loader)

        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "softmax_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(classifier, f)
        print(f"[SoftmaxClassifier] Modello salvato in {model_path}")

    # --- Testing ---
    elif mode == "test":
        model_path = os.path.join("models", "softmax_model.pkl")

        classifier = SoftmaxClassifier(
            embedding_dim=embeddings.shape[1],
            num_classes=len(np.unique(labels))
        ).to(device)

        state_dict = torch.load(model_path, map_location=device)
        classifier.load_state_dict(state_dict)
        classifier.eval()

        X_t = torch.tensor(embeddings, dtype=torch.float)
        y_t = torch.tensor(labels, dtype=torch.long)
        test_ds = TensorDataset(X_t, y_t)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        val_loss, val_acc, _ = classifier.test(test_loader)
        print(f"[SoftmaxClassifier] Test -> loss: {val_loss:.4f}, acc: {val_acc:.2f}%")


if __name__ == "__main__":
    train_softmax_classifier(mode="train")