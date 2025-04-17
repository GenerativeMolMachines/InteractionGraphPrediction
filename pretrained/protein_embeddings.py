from transformers import BertTokenizerFast, BertModel
from sentence_transformers import SentenceTransformer


class BertProtein:
    def __init__(self, path: str = 'unikei/bert-base-proteins'):
        self.path = path
        self.tokenizer = BertTokenizerFast.from_pretrained(self.path)
        self.model = BertModel.from_pretrained(self.path)

    def get_emb(self, seq: str, idx: int = 0):
        tokens = self.tokenizer(seq, return_tensors='pt')
        predictions = self.model(**tokens)
        return predictions[idx]


class ProteinEmbeddings:
    def __init__(self, path: str ='monsoon-nlp/protein-matryoshka-embeddings'):
        self.model = SentenceTransformer(path)

sequences = ["M S L E Q K...", "M A R N W S F R V..."]

model = SentenceTransformer('monsoon-nlp/protein-matryoshka-embeddings')
embeddings = model.encode(sequences)
print(embeddings)
