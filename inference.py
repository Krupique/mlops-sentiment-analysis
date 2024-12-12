# Main training script
import yaml
import argparse
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import pandas as pd

import transformers
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertModel

import torch.optim as optim
import torch.nn as nn

from data.dataset import TextClassificationDataset
from utils.preprocessing import encode, to_categorical
from models import classifier
from utils.trainer import Trainer


def main(sentence, config_path):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Tokenizer
    fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase = False)
    
    # Load Label Encoder Weights
    le = joblib.load('label_encoder_weights.joblib')

    # Load Model
    model = torch.load("models/model_v1_complete.pth")
    # Putting the model in evaluation mode if it is for inference
    model.eval()

    # Create a dataframe with the sentence
    df_new = pd.DataFrame({'text': [sentence]})
    df_new = TextClassificationDataset(df_new).data

    new_data = encode(df_new['transformed_text'], fast_tokenizer, max_len = config['train']['max_length'])
    # Converting new_data to a PyTorch tensor if it isn't already
    new_data_tensor = torch.tensor(new_data).to(device)
    model = model.to(device)

    # Prediction
    with torch.no_grad():
        prediction = model(new_data_tensor)

    # Predicted labels (choosing the class index with highest probability)
    predicted_label = torch.argmax(prediction, dim=1).cpu().numpy()

    # Get the class name
    class_name = le.inverse_transform(predicted_label)
    print(class_name)

    return class_name
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis using BERT")
    parser.add_argument(
        "--sentence",
        type=str,
        required=True,
        help="Write a sentence to be classified"
    )
    args = parser.parse_args()

    config_path = "config/config.yaml"
    
    response = main(args.sentence, config_path)