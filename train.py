# Main training script
import yaml
import torch

from data.dataset import TextClassificationDataset
from transformers import AutoTokenizer


def main(config_path="config/config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Device setp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device name: {device}')

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_name"])
    dataset = TextClassificationDataset(config['data']['train_dataset_path'], tokenizer)

    print(dataset)



if __name__ == "__main__":
    main()