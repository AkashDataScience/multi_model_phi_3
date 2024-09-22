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
BATCH_SIZE = 4
MAX_STEPS = 3000
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
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
)

# Initialize loss function
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.train()

for step in range(MAX_STEPS):
    print(f"Step {step + 1}/{MAX_STEPS}")
    epoch_loss = 0
    valid_batches = 0

    
    image_name, input_ids, target_ids = next(iter(dataloader))
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)


    optimizer.zero_grad()


    outputs = model(image_name, input_ids)

    # Select the logits for all text tokens after the 5 separator tokens
    text_token_logits = outputs.logits[:, 1:, :]  # Start from index 5 to skip separator tokens

    # Flatten the logits and target sequence for loss calculation
    text_token_logits_flat = text_token_logits.reshape(-1, text_token_logits.size(-1))
    target_ids_flat = target_ids.reshape(-1)

    # Calculate loss over the text token sequence
    loss = criterion(text_token_logits_flat, target_ids_flat)
    
    loss.backward()
    optimizer.step()
    scheduler.step()

    epoch_loss += loss.item()
    valid_batches += 1

    if step % 100 == 0:
        print(f"Loss: {epoch_loss/valid_batches}")

# Save the trained model
torch.save(model.state_dict(), 'trained_clip_phi3_model.pth')
print("Training completed and model saved.")
