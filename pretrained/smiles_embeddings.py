from transformers import BertTokenizerFast, BertModel


class BertSmiles:
    def __init__(self, path: str = 'unikei/bert-base-smiles'):
        self.path = path
        self.tokenizer = BertTokenizerFast.from_pretrained(self.path)
        self.model = BertModel.from_pretrained(self.path)

    def get_emb(self, seq: str, idx: int = 0):
        tokens = self.tokenizer(seq, return_tensors='pt')
        predictions = self.model(**tokens)
        return predictions[idx]