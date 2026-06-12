import torch
import torch.nn as nn

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels, body, texts in loader:
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)
        body = body.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(images, body, texts)
        loss = criterion(outputs, labels)
        
        # Backward Pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, body, texts in loader:
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)
            body = body.to(device)
            
            outputs = model(images, body, texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    acc = correct / total
    return total_loss / len(loader), acc
