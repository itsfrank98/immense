import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm   # barra di progresso utile per monitorare

class WordEmb:
    def __init__(self, model_name="sarkerlab/SocBERT-base", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True, config=config)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def texts_to_vec(self, texts, batch_size=16, max_length=128):
        """
        Calcola gli embedding per una lista di testi in batch.
        Gestisce automaticamente la memoria GPU e fallback su CPU.
        """
        self.model.eval()  # Disattiva dropout, batchnorm, ecc.
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Calcolo embedding"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Media lungo la dimensione della sequenza (dim=1)
                    batch_emb = outputs.last_hidden_state.mean(dim=1).detach()
                    # Sposta su CPU per liberare memoria GPU
                    embeddings.append(batch_emb.cpu())
                # Libera memoria GPU ogni batch
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[WARNING] Out of memory al batch {i // batch_size}. Passo a CPU.")
                    torch.cuda.empty_cache()
                    self.device = torch.device("cpu")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        batch_emb = outputs.last_hidden_state.mean(dim=1)
                        embeddings.append(batch_emb.cpu())
                else:
                    raise e

        # Concatena tutti gli embedding finali
        return torch.cat(embeddings, dim=0)

    def embeddings_from_csv(self, csv_path, batch_size=16, max_length=128):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["id", "text_cleaned", "label"])  # sicurezza

        # unisco i testi per ogni utente
        user_text_dict = df.groupby("id")["text_cleaned"].apply(lambda x: " ".join(x.astype(str).tolist())).to_dict()
        ids = list(user_text_dict.keys())
        texts = list(user_text_dict.values())

        # calcolo embedding in batch
        input_tensor = self.texts_to_vec(texts, batch_size=batch_size, max_length=max_length)

        # etichette
        labels_dict = df.drop_duplicates("id").set_index("id")["label"].to_dict()
        label_tensor = torch.tensor([labels_dict[uid] for uid in ids])

        # 🔹 Salvataggio su file
        torch.save(input_tensor, "dataset/embeddings.pt")
        torch.save(label_tensor, "dataset/labels.pt")

        print(f"[WordEmb] Salvati {len(ids)} embeddings in dataset/embeddings.pt e labels in dataset/labels.pt")

        return input_tensor, label_tensor
