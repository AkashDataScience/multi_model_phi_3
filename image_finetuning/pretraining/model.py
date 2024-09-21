import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class Projections(nn.Module):
    def __init__(self, clip_embed, phi_embed):
        super().__init__()
        self.projection = nn.Linear(clip_embed, phi_embed)
    
    def forward(self, x):
        return self.projection(x)

class ClipPhi3Model(nn.Module):
    def __init__(self, model_name, clip_embed, phi_embed):
        super().__init__()
        self.phi = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                        trust_remote_code=True)
        self.projections = Projections(clip_embed, phi_embed)
        
        # Load image embeddings
        self.image_embeddings = torch.load('clip_embeddings.pt')
        
        # Freeze phi model weights
        for param in self.phi.parameters():
            param.requires_grad = False
    
    def forward(self, image_ids, input_ids):
        # Apply embeddings to input_ids
        text_embeds = self.phi.get_input_embeddings()(input_ids)
        
        # Load image embeddings
        image_embeds = torch.stack([self.image_embeddings[id] for id in image_ids])
        
        # Apply projection to image embeddings
        projected_image_embeds = self.projections(image_embeds)
        
        # Combine image and text embeddings
        combined_embeds = torch.cat([projected_image_embeds, text_embeds], dim=1)
        
        # Pass through phi-3 model
        outputs = self.phi(inputs_embeds=combined_embeds).logits
        
        return outputs

# Usage example:
# model = ClipPhi3Model("microsoft/phi-2", clip_embed=512, phi_embed=2560)
# outputs = model(image_ids=['image1', 'image2'], input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]))
