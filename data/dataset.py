# Data Processing and dataset class
import pandas as pd
from torch.utils.data import Dataset
import spacy

class TextClassificationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.data = pd.read_csv(csv_path, header=None, delimiter=';')
        self.data = self.data.rename(columns={0: 'text', 1: 'feeling'})

        self.spacy_nlp = spacy.load('en_core_web_md')

    def data_preprocessing(self, text):
        doc = self.spacy_nlp(text)
        tokens = [token.lemma_.lower().strip() for token in doc if not token.is_stop]
        return ' '.join(tokens)