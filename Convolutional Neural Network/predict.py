from src.model import CustomCNN
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

def prediction():
    weights = torch.load("model/weights_best.pth", map_location="cuda")
    config = torch.load("model/configs.pth", map_location="cuda")

    model = CustomCNN().to(device)
    model.load_state_dict(weights)
    model = model.to(device)
    
    with torch.no_grad():
        model.eval()
        output = model(feature)
        preds = output.argmax(1)

    fig, axes = plt.subplots(6, 6, figsize=(24, 24))
    for img, label, pred, ax in zip(feature, target, preds, axes.flatten()):
        ax.imshow(img.permute(1, 2, 0).cpu())
        font = {"color": 'r'} if label != pred else {"color": 'g'}
        label, pred = label2cat[label.item()], label2cat[pred.item()]
        ax.set_title(f"Label: {label} | Pred: {pred}", fontdict=font);
        ax.axis('off');

