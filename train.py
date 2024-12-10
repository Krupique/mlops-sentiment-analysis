# Main training script
import yaml
import torch
from sklearn.model_selection import train_test_split

import transformers
from tokenizers import BertWordPieceTokenizer

from data.dataset import TextClassificationDataset
from utils.preprocessing import encode


def main(config_path="config/config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Device setp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device name: {device}')

    # Load tokenizer and dataset
    print('Load tokenizer')
    tokenizer_bert = transformers.DistilBertTokenizer.from_pretrained(config["model"]["pretrained_name"])
    # Save the tokenizer and the vocabulary locally
    tokenizer_bert.save_pretrained('.')
    # Load a faster tokenizer using the vocabulary of main tokenizer 
    fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase = False)
    
    dataset_train = TextClassificationDataset(config['data']['train_dataset_path'])
    dataset_test = TextClassificationDataset(config['data']['test_dataset_path'])

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

    print(X_final_train)



if __name__ == "__main__":
    main()