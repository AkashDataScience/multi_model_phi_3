import torch
from torch.utils.data import Dataset
import random

class ImageConversationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_name = item['id']
        conversations = item['conversations']

        # Randomly select an even-numbered index
        max_even_index = len(conversations) - 1
        selected_index = random.randrange(0, max_even_index + 1, 2)

        # Get the human question and GPT response
        human_msg = conversations[selected_index]['value']
        ai_msg = conversations[selected_index + 1]['value']

        # Tokenize the conversation
        tokenized = self.tokenizer(
            human_msg,
            ai_msg,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'image_name': image_name,
            'input_ids': tokenized.input_ids.squeeze(),
            'attention_mask': tokenized.attention_mask.squeeze()
        }
