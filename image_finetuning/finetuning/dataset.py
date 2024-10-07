import torch
from torch.utils.data import Dataset
import random

class ImageConversationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_name = item['id']
        conversations = item['text']

        conversations_tokens = self.tokenizer(conversations, max_length=self.max_length,
                                              truncation=True, padding='max_length',
                                              return_tensors='pt')
        
        conversations_ids = conversations_tokens['input_ids']
        conversations_mask = conversations_tokens['attention_mask']

        labels = conversations_ids.clone()
        labels[:, :-1] = conversations_ids[:, 1:]
        
        return {"image_name": image_name, "conversations_ids": conversations_ids,
                "conversations_mask": conversations_mask, "labels": labels}