import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from model import ClipPhi3Model
from dataset import ImageConversationDataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Constants
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
CLIP_EMBED = 512  # Adjust based on your CLIP model
PHI_EMBED = 3072  # Adjust based on the phi-3-mini model
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100

# Load dataset
dataset = load_dataset('json', data_files='llava_instruct_150k.json', split='train')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Create ImageConversationDataset
image_dataset = ImageConversationDataset(dataset, tokenizer)  # Pass None as tokenizer since we're not using it here

# Create DataLoader
dataloader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = ClipPhi3Model(MODEL_NAME, CLIP_EMBED, PHI_EMBED)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize optimizer
optimizer = AdamW(model.projections.parameters(), lr=LEARNING_RATE)

# Initialize scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(dataloader) * EPOCHS
)

# Initialize loss function
criterion = torch.nn.CrossEntropyLoss()

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.train()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    epoch_loss = 0
    valid_batches = 0

    for batch in tqdm(dataloader):
        image_ids = batch['image_name']
        input_ids = batch['input_ids'].to(device).to(torch.bfloat16)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        logits = model(image_ids, input_ids)

        # Shift the input_ids and logits for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Calculate loss
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        valid_batches += 1

    if valid_batches > 0:
        avg_loss = epoch_loss / valid_batches
        print(f"Average loss: {avg_loss}")
    else:
        print("No valid batches in this epoch.")

# Save the trained model
torch.save(model.state_dict(), 'trained_clip_phi3_model.pth')
print("Training completed and model saved.")
