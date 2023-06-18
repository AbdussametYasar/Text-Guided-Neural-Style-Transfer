from __future__ import print_function
import numpy as np
import matplotlib as mpl
import cv2
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import os
from sklearn.cluster import KMeans
from scipy.spatial import distance

def ensure_same_size(image1_path, image2_path):
    # Load images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Check if sizes are the same
    if image1.size != image2.size:
        # Resize image2 to match image1
        image2 = image2.resize(image1.size)

    return image1, image2

# Load the data
train_features = np.load("train_features.npy")
train_labels = np.load("train_labels.npy")

# Load the images
image_dir = "C:/Users/Samet/Desktop/perseptron/Images/Train"
dataset = []
image_files = []
for subdir, dirs, files in os.walk(image_dir):
    for file in sorted(files):
        img = cv2.imread(os.path.join(subdir, file))
        if img is not None:
            dataset.append(img)
            image_files.append(file)
dataset = np.array(dataset, dtype=object)

# Create a dictionary to hold the kmeans models for each label
kmeans_models = {}
unique_labels = np.unique(train_labels)

print("Choose a style: anime, digital, oil, pixel")
style_choice = input()

# Print unique labels in train_labels
print("Unique labels in train_labels: ", np.unique(train_labels))

# Map style choices to labels
style_map = {"anime": "anime_faces", "digital": "digital_art", "oil": "oil_painting", "pixel": "pixel_art"}

# Print style_choice and corresponding label
print("Style choice: ", style_choice)
label = style_map[style_choice]
print("Corresponding label: ", label)

# Convert both arrays to the same type before comparison
train_labels_str = np.array(train_labels, dtype=str)
label_str = str(label)

# print(f"Debug info: {np.unique(train_labels_str)}")  # Debug print

# Create a boolean mask
mask = train_labels_str == label_str

# Check if there is at least one True in the mask
if np.any(mask):
    label_features = train_features[mask]
    label_features = label_features.reshape(label_features.shape[0], -1)
else:
    print(f"No matches found for label {label_str} in train_labels.")


# Create and fit a KMeans model to these features
kmeans = KMeans(n_clusters=7)  # Adjust the number of clusters as needed
kmeans.fit(label_features)

# For each cluster center
for center in kmeans.cluster_centers_:
    # Compute distances to the cluster center
    distances = distance.cdist([center], label_features, 'euclidean')[0]

    # Get the index of the closest image
    closest_image_idx = np.argmin(distances)

    # This is your representative 'style' image for this cluster
    representative_image = dataset[train_labels == label][closest_image_idx]

    # Convert the representative image from BGR to RGB (since cv2.imread loads images in BGR format)
    representative_image = cv2.cvtColor(representative_image, cv2.COLOR_BGR2RGB)

    # Convert numpy array to PIL image
    representative_image = Image.fromarray(representative_image)

    # Save the representative image
    representative_image.save(f"{style_choice}_representative_image.jpg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize(imsize),  
    transforms.ToTensor()])  


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader(f"{style_choice}_representative_image.jpg")
content_img = image_loader("C:/Users/Samet/Desktop/Perseptron Ağlar Proje/images/face.jpg")

if style_img.size() != content_img.size():
    image1, image2 = ensure_same_size(content_img, style_img)


unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  
    image = image.squeeze(0)      
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

# plt.figure()
# imshow(style_img, title='Style Image')

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
def gram_matrix(input):
    a, b, c, d = input.size()  
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  

    G = torch.mm(features, features.t())  

    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)


    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

input_img = content_img.clone()

plt.figure()
imshow(input_img, title='Input Image')

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=750,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

plt.ioff()
plt.show()

input("Press Enter to continue...")