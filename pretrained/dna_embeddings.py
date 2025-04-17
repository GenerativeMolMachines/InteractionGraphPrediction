import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM


class BertDNA:
    def __init__(self, path: str = "zhihan1996/DNABERT-2-117M"):
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True)

    def get_emb(self, seq: str, pooling_type: str = "max"):
        inputs = self.tokenizer(seq, return_tensors='pt')["input_ids"]
        hidden_states = self.model(inputs)[0]  # [1, sequence_length, 768]

        embedding = {"max": torch.max(hidden_states[0], dim=0)[0], "mean": torch.mean(hidden_states[0], dim=0)}
        return embedding[pooling_type]


class BertDNAPlant:
    def __init__(self, path: str = 'zhangtaolab/plant-dnabert-BPE'):
        self.model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    def get_emb(self, seq: str):
        tokens = self.tokenizer(seq, padding="longest")['input_ids']
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        inputs = self.tokenizer(seq, truncation=True, padding='max_length', max_length=512,
                           return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outs = self.model(
            **inputs,
            output_hidden_states=True
        )
        return outs['hidden_states'][-1].detach().numpy()
