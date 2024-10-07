import torch
from torch.utils.data import Dataset
import random
import numpy as np

class ImageConversationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        # Load image embeddings and convert to tensors if necessary
        self.image_embeddings = torch.load('clip_embeddings.pt')

        for key, value in self.image_embeddings.items():
            if isinstance(value, np.ndarray):
                self.image_embeddings[key] = torch.from_numpy(value).to(torch.bfloat16)
            else:
                self.image_embeddings[key] = value.to(torch.bfloat16)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_name = item['id']
        conversations = item['text']

        image_embeds = self.image_embeddings[image_name]
        
        return {"image_embeds": image_embeds, "conversations": conversations}