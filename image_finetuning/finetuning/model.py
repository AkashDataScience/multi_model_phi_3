import torch
import torch.nn as nn
from transformers import PreTrainedModel
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
class Projections(nn.Module):
    def __init__(self, clip_embed, phi_embed, num_projection_layers=6):
        super().__init__()

        self.output = nn.Linear(clip_embed, phi_embed)
        self.norm = nn.LayerNorm(phi_embed)
        self.projection_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(phi_embed, phi_embed),
                    nn.GELU(),  
                    nn.Linear(phi_embed, phi_embed),
                )
                for _ in range(num_projection_layers)
            ]
        )

    def forward(self, x):
        x = self.output(x)
        x = self.norm(x)
        for layer in self.projection_layers:
            residual = x
            x = layer(x) + residual 
        
        return x

class ClipPhi3Model(PreTrainedModel):
    def __init__(self, phi_model, clip_embed, phi_embed, projection_path=None):
        super().__init__(phi_model.config)
        self.phi = phi_model
        self.projections = Projections(clip_embed, phi_embed)
        if projection_path:
            self.projections.load_state_dict(torch.load(projection_path, map_location=torch.device(device)), strict=False)
        
        # Load image embeddings and convert to tensors if necessary
        self.image_embeddings = torch.load('clip_embeddings.pt')
        for key, value in self.image_embeddings.items():
            if isinstance(value, np.ndarray):
                self.image_embeddings[key] = torch.from_numpy(value).to(torch.bfloat16)
            else:
                self.image_embeddings[key] = value.to(torch.bfloat16)
        
        # Convert projections to bfloat16
        self.projections.to(torch.bfloat16)

    def forward(self, image_ids, conversations_ids, conversations_mask, labels):
        # Apply embeddings to input_ids
        text_embeds = self.phi.get_input_embeddings()(conversations_ids)
        
        # Load image embeddings
        image_embeds = torch.stack([self.image_embeddings[id] for id in image_ids]).to(device)
        
        # Apply projection to image embeddings
        projected_image_embeds = self.projections(image_embeds)
        
        # Combine image and text embeddings
        combined_embeds = torch.cat([projected_image_embeds, text_embeds], dim=1)
        combined_mask = torch.cat([torch.ones((projected_image_embeds.shape)), text_embeds], dim=1)
        
        # Pass through phi-3 model
        outputs = self.phi(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=labels, return_dict=True)
        
        return outputs
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.phi.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.phi.gradient_checkpointing_disable()
        
