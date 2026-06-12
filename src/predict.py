import torch
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def run_evaluation_pipeline(model, val_loader, device):
    """
    Evaluates the model across the entire validation loader to collect
    raw outputs, predictions, and ground-truth values for paper analysis.
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    for image, label, body, text in tqdm(val_loader, desc="Running Pipeline"):
        image = image.to(device)
        body = body.to(device)
        label = label.to(device)
        
        # Run inference
        outputs = model(image, body, text)
        probs = torch.sigmoid(outputs).squeeze()
        preds = (probs > 0.5).int()
        
        # Store results safely as arrays
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)
