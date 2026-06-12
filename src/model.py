import torch
import torch.nn as nn
import timm
from transformers import AutoModel

class MultimodalRadiologyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. IMAGE ENCODER (EfficientNet-B0)
        self.image_encoder = timm.create_model(
            "efficientnet_b0", 
            pretrained=True, 
            num_classes=0
        )
        img_dim = self.image_encoder.num_features # 1280 features
        
        # 2. TEXT ENCODER (Bio_ClinicalBERT)
        self.text_encoder = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        txt_dim = 768
        
        # Freeze BERT weights to prevent overfitting during verification
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # 3. CLINICAL TABULAR STRUCTURAL ENCODER
        self.body_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # 4. MULTIMODAL FUSION CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Linear(img_dim + txt_dim + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, image, body, texts, tokenizer):
        """
        Args:
            image: Batch of preprocessed radiology images
            body: Tabular clinical structure tensors (anatomical region markers)
            texts: Raw clinical text string list/tuple
            tokenizer: AutoTokenizer tracking instance from training script
        """
        # Extract visual engineering representations
        img_feat = self.image_encoder(image)
        
        # Tokenize text batches dynamically on the correct running device
        tokens = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(image.device)
        
        # Extract linguistic representations using the [CLS] embedding
        txt_feat = self.text_encoder(**tokens).last_hidden_state[:, 0]
        
        # Process structural domain vector elements
        body_feat = self.body_encoder(body)
        
        # Concatenate multi-source embeddings along feature dimension
        x = torch.cat([img_feat, txt_feat, body_feat], dim=1)
        
        return self.classifier(x)
