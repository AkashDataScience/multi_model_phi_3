import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import os

class ImageConversationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_name = item['image']
        conversations = item['conversations']

        # Randomly select an even-numbered index
        max_even_index = len(conversations)
        selected_index = random.randrange(0, max_even_index, 2)

        # Get the human question and GPT response
        human_msg = conversations[selected_index]['value']
        ai_msg = conversations[selected_index + 1]['value']

        input_ids = self.tokenizer.encode(human_msg).ids
        target_ids = self.tokenizer.encode(ai_msg).ids


        return image_name, input_ids, target_ids
