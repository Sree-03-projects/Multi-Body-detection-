import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# Mapping for the clinical tabular structural features
BODY_MAP = {
    "extremity": [1, 0, 0],
    "spine": [0, 1, 0],
    "chest": [0, 0, 1]
}

class RadiologyDataset(Dataset):
    def __init__(self, df, transform, root_map):
        """
        Args:
            df: Pandas DataFrame containing ['image', 'label', 'domain']
            transform: Albumentations preprocessing pipeline
            root_map: Dictionary mapping domains to their image folder paths
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.root_map = root_map
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Resolve image path dynamically based on clinical domain
        img_path = os.path.join(self.root_map[row.domain], row.image)
        
        # Load and convert image to RGB numpy array for albumentations
        image = np.array(Image.open(img_path).convert("RGB"))
        image = self.transform(image=image)["image"]
        
        # Format metrics targets
        label = torch.tensor(row.label, dtype=torch.float32)
        body = torch.tensor(BODY_MAP[row.domain], dtype=torch.float32)
        text = f"Radiology examination of {row.domain}"
        
        return image, label, body, text
