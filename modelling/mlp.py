from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from kornia.losses import binary_focal_loss_with_logits
from torch.nn import Linear
from torch.nn.functional import relu, nll_loss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import save_to_pickle


class MLP(torch.nn.Module):
    def __init__(self, X_train, y_train, model_path, loss, batch_size=64, epochs=50, weights=None, hidden_dim=3):
        super(MLP, self).__init__()
        self.X_train = X_train
        self.weights = weights
        self.y_train = torch.tensor(y_train, dtype=torch.float)
        self._model_path = join(model_path)
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss

        input_dim = X_train.shape[1]
        num_classes = len(torch.unique(self.y_train.long()))  # numero di classi dinamico

        print(f"[MLP init] input_dim={input_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}")

        # definizione rete
        self.input = Linear(in_features=input_dim, out_features=hidden_dim)
        self.output = Linear(in_features=hidden_dim, out_features=num_classes)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.weights is not None:
            self.weights = self.weights.to(self.device)
        self.to(self.device)

    def forward(self, x):
        x = self.input(x)
        x = relu(x)
        out = self.output(x)
        if self.loss == "weighted":
            out = torch.log_softmax(out, dim=-1)
        return out

    def train_mlp(self, optimizer):
        print("Training MLP...")
        ds = TensorDataset(self.X_train, self.y_train)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        best_loss = 9999
        for epoch in range(self.epochs):
            self.train()
            total_loss = 0

            for batch_x, batch_y in tqdm(dl):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                out = self(batch_x)
                if self.loss == "focal":
                    t = [int(el) for el in batch_y]
                    target = torch.tensor(np.eye(2, dtype='uint8')[t], dtype=torch.float).to(self.device)
                    loss = binary_focal_loss_with_logits(out, target=target, reduction="mean")
                elif self.loss == "weighted":
                    loss = nll_loss(out, batch_y.long(), weight=self.weights)
                elif self.loss == "none":
                    loss = F.cross_entropy(out, batch_y.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            l = total_loss/self.batch_size
            if epoch % 5 == 0:
                print("\nEpoch: {}, Loss: {}".format(epoch, l))
            if l < best_loss:
                best_loss = l
                print("New best model found at epoch {}. Loss: {}".format(epoch, best_loss))
                save_to_pickle(self._model_path, self)

    def predict_proba(self, X_test):
        self.eval()
        with torch.no_grad():
            logits = self(torch.tensor(X_test, dtype=torch.float).to(self.device))
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def test(self, X_test):
        probs = self.predict_proba(X_test)
        return np.argmax(probs, axis=-1)
