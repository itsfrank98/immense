import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from kornia.losses import binary_focal_loss_with_logits


# --------------------------------------------------
# Classe SoftmaxClassifier
# --------------------------------------------------
VERBOSITY = False
class SoftmaxClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, loss="weighted", weights=None):
        """
        Classificatore Softmax dinamico.
        """
        super(SoftmaxClassifier, self).__init__()
        self.device = torch.device("cpu")
        self.loss = loss  # ✅ mancava questa riga!
        self.weights = weights.to(self.device) if weights is not None else None

        # Architettura semplice
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
        """
        Ciclo di training del classificatore softmax.
        """
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

            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=not VERBOSITY):
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
                    if VERBOSITY:
                        print(f"New best model found at epoch {epoch + 1}. Val Loss: {val_loss:.4f}")

            if epoch % 5 == 0:
                print(f"\nEpoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
                if val_loader is not None and VERBOSITY:
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
