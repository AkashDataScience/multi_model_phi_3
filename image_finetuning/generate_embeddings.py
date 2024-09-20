import torch
from PIL import Image
import clip
import os
from tqdm import tqdm
from datasets import load_dataset

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load Instruct 150k dataset
dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")

# Process images and store embeddings
embeddings = {}
for item in tqdm(dataset):
    image_path = item['image']
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
    
    embeddings[item['id']] = image_features.cpu().numpy()

# Save embeddings
torch.save(embeddings, 'clip_embeddings.pt')
