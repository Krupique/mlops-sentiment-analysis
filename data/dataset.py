# Data Processing and dataset class
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextClassificationDataset():
    def __init__(self, data):
        # Ensure required NLTK data is downloaded
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        
        self.data = data
        self.data = self.data.rename(columns={0: 'text', 1: 'feeling'})
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.data['transformed_text'] = self.data['text'].apply(self.custom_nlp)
        print('Preprocessing completed')    

    def custom_nlp(self, text):
        # Tokenize the text
        tokens = re.findall(r'\b\w+\b', text.lower())  # Extract words, ignoring punctuation
        # Remove stopwords, remove punctuation, and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token.strip()) for token in tokens 
            if token not in self.stop_words and token.isalnum()  # Check for alphanumeric tokens
        ]
        # Join tokens into a single string
        return ' '.join(processed_tokens)