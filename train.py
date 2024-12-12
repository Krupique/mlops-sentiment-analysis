# Main training script
import json
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


def main(config_path="config/config.json"):
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device name: {device}')

    # Load tokenizer and dataset
    print('Load tokenizer')
    tokenizer_bert = transformers.DistilBertTokenizer.from_pretrained(config["model"]["pretrained_name"])
    # Save the tokenizer and the vocabulary locally
    tokenizer_bert.save_pretrained('.')
    # Load a faster tokenizer using the vocabulary of main tokenizer 
    fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase = False)
    
    print('Data Preprocessing')
    df_train_raw = pd.read_csv(config['data']['train_dataset_path'], header=None, delimiter=';')
    df_test_raw = pd.read_csv(config['data']['test_dataset_path'], header=None, delimiter=';')

    dataset_train = TextClassificationDataset(df_train_raw)
    dataset_test = TextClassificationDataset(df_test_raw)

    print('df train')
    df_train = dataset_train.data    
    df_test = dataset_test.data
    print(df_train.head()) 

    # Data Splitting
    print('data splitting')
    X_train, X_valid, y_train, y_valid = train_test_split(df_train['transformed_text'].values,
                                                            df_train['feeling'].values,
                                                            test_size = config['data']['train_split'],
                                                            random_state = config['data']['random_seed'],
                                                            stratify = df_train['feeling'])

    print('data encode')
    X_final_train = encode(X_train, fast_tokenizer, max_len = config['train']['max_length'])
    X_final_valid = encode(X_valid, fast_tokenizer, max_len = config['train']['max_length'])
    X_final_test = encode(df_test['transformed_text'].to_numpy(), fast_tokenizer, max_len = config['train']['max_length'])

    # Define the encoder of output data
    le = LabelEncoder()
    # Applying the label encoder (fit_transform only on train data)
    y_train_le = le.fit_transform(y_train)
    y_valid_le = le.transform(y_valid)
    y_test_le = le.transform(df_test['feeling'])
    joblib.dump(le, 'label_encoder_weights.joblib')

    # Convert the output variable to categorical
    y_train_encoded = to_categorical(y_train_le)
    y_valid_encoded = to_categorical(y_valid_le)
    y_test_encoded = to_categorical(y_test_le)

    # Prepare the dataset in the expected format of Pytorch
    train_dataset = TensorDataset(torch.tensor(X_final_train), torch.tensor(y_train_encoded))
    train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=True)

    valid_dataset = TensorDataset(torch.tensor(X_final_valid), torch.tensor(y_valid_encoded))
    valid_loader = DataLoader(valid_dataset, batch_size=config['model']['batch_size'])

    test_dataset = TensorDataset(torch.tensor(X_final_test), torch.tensor(y_test_encoded))
    test_loader = DataLoader(test_dataset, batch_size=config['model']['batch_size'], shuffle=True)

    # Creates an instance of the pre-trained, multilingual DistilBERT model suitable for use with PyTorch
    transformer_model = DistilBertModel.from_pretrained(config['model']['pretrained_name'])


    model = classifier.Model(transformer=transformer_model)
    # Model summary
    print(model)
    trainer = Trainer(model=model, train_loader=train_loader, valid_loader=valid_loader, config=config, device=device)
    trainer.train()


    # Model evaluation
    model.eval()

    # Converting X_final_test to a PyTorch tensor
    X_test_final_tensor = torch.tensor(X_final_test).to(device)

    # Predictions
    with torch.no_grad():
        predictions = model(X_test_final_tensor)

    # Predicted labels (choosing the class index with highest probability)
    predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()

    print(classification_report(y_test_le, predicted_labels))
    print(confusion_matrix(y_test_le, predicted_labels))
    print(accuracy_score(y_test_le, predicted_labels))


    torch.save(model, "models/model_v1_complete.pth")


if __name__ == "__main__":
    main()