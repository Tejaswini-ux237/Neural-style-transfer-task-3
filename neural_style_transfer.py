# Neural Style Transfer - Task 3 for CODTECH Internship
# Author: UPPALA TEJASWINI

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Image loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image_name, max_size=400, shape=None):
    image = Image.open(image_name).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    image = in_transform(image).unsqueeze(0)
    return image.to(device)

# Load VGG19 model
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # content representation
                  '28': 'conv5_1'}

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

# Gram matrix for style representation
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram 

# Run style transfer
def run_style_transfer(content_path, style_path, output_path, steps=2000, style_weight=1e6, content_weight=1):
    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    content = image_loader(content_path)
    style = image_loader(style_path, shape=[content.size(2), content.size(3)])

    target = content.clone().requires_grad_(True).to(device)

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    optimizer = optim.Adam([target], lr=0.003)

    for step in range(steps):
        target_features = get_features(target, vgg)

        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        style_loss = 0
        for layer in style_grams:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}, Total loss: {total_loss.item():.4f}")

    final_img = target.cpu().clone().squeeze(0)
    final_img = final_img.detach()

    # Unnormalize
    final_img = final_img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    final_img = final_img + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    final_img = final_img.clamp(0, 1)

    # Save result
    to_pil = transforms.ToPILImage()
    to_pil(final_img).save(output_path)
    print(f"Styled image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    content_image = "content.jpg"
    style_image = "style.jpg"
    output_image = "output.jpg"

    if os.path.exists(content_image) and os.path.exists(style_image):
        run_style_transfer(content_image, style_image, output_image)
    else:
        print("Please place 'content.jpg' and 'style.jpg' in the same folder as this script.")
