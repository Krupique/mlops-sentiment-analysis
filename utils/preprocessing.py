import numpy as np
from tqdm import tqdm

# Function to encode the text into a sequence of integers for input to the BERT model
def encode(texts, tokenizer, chunk_size = 256, max_len = 512):

    # Enable truncation in the tokenizer to a specified max length
    tokenizer.enable_truncation(max_length = max_len)

    # Enable padding in th tokenizer to a specified max length
    tokenizer.enable_padding(length = max_len)

    # Initialize a list to stor the encoded IDs
    all_ids = []

    # Iterate over texts in chunk of size 'chunk_size'
    for i in tqdm(range(0, len(texts), chunk_size)):

        # Create a chunk of text
        text_chunk = texts[i:i+chunk_size].tolist()

        # Encode the chunk of text in batch
        encs = tokenizer.encode_batch(text_chunk)

        # Extend the list 'all_ids' with the encoded IDs
        all_ids.extend([enc.ids for enc in encs])

    # Return the IDs list as an array numpy
    return np.array(all_ids)


def to_categorical(y, num_classes=None, dtype='float32'):
    # Converts y into an array of numpy
    y = np.array(y, dtype='int')
    
    # If the number of classes is not given, it will be determined from y
    if not num_classes:
        num_classes = np.max(y) + 1
    
    # Initialize the output array with zeros
    categorical = np.zeros((y.shape[0], num_classes), dtype=dtype)
    
    # Fill the matrix with 1 in the position corresponding to the class of each label
    categorical[np.arange(y.shape[0]), y] = 1
    
    return categorical
