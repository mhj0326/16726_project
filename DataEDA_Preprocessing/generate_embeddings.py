from cxrclip.evaluator import Evaluator
from cxrclip.model import build_model
import os
import glob
import random
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys 
import numpy as np
#GENERATE IMAGE LIST TO EXTRACT EMBEDDINGS

# # Base directory containing the subdirectories
# base_dir = "/home/younjoon/data/dataset/chestx-ray14/"

# # Pattern to match subdirectories starting with 'images_'
# subdir_pattern = os.path.join(base_dir, "images_*")

# # Final list to hold the randomly selected png files
# final_png_list = []

# # Iterate through each subdirectory that matches the pattern
# for subdir in glob.glob(subdir_pattern):
#     # Find all png files in the current subdirectory
#     png_files = glob.glob(os.path.join(subdir+'/images/', "*.png"))
    
#     # Check if there are at least 800 png files
#     if len(png_files) >= 800:
#         # Randomly select 800 png files
#         selected_pngs = random.sample(png_files, 800)
#     else:
#         # If less than 800, take all of them
#         selected_pngs = png_files
    
#     # Add the selected png files to the final list
#     final_png_list.extend(selected_pngs)

# # Now final_png_list contains up to 800 randomly selected png files from each subdirectory
# print(f"Total selected png files: {len(final_png_list)}")
# print(final_png_list[0])
# # Define the path for the .txt file where the list will be saved
# save_path = "selected_png_list.txt"

# # Write the list to the file
# with open(save_path, 'w') as file:
#     for png_path in final_png_list:
#         file.write(png_path + '\n')

# print(f"List of PNG files saved to {save_path}")

class PNGDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def encode_image(model, image: torch.Tensor):
    device = 'cuda'
    with torch.no_grad():
        img_emb = model.encode_image(image.to(device))
        img_emb = model.image_projection(img_emb) if model.projection else img_emb
        img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
    return img_emb.detach().cpu().numpy()

def evaluate_clip(model, dataloader):
       
        #log.info(f"Load model {checkpoint}")
        #ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(torch.load('/home/rwiddhi/rwiddhi/checkpoint/r50_mcc.tar'), strict=False)
        model.eval()


        image_embeddings = []
        text_embeddings = []
        texts = []
        label_names = []
        label_indices = []
        for batch in tqdm(dataloader):
            img_emb = encode_image(model, batch)
            image_embeddings.append(img_emb)

        image_embeddings = np.concatenate(image_embeddings, axis=0)

        return image_embeddings
#Load list of image_paths
# Define the path to your text file
file_path = '/home/rwiddhi/selected_png_list.txt'  # Replace this with the actual file path

# Open the text file and read the contents into a list
with open(file_path, 'r') as file:
    # Use readlines() to read all lines at once
    lines = file.readlines()
    
    # Strip newline characters from each line
    paths_list = [line.strip() for line in lines]

# Now, paths_list contains all the paths from the text file without newline characters


#Load config
with open('/home/rwiddhi/config.json', 'r') as f:
    config = json.load(f)

config_model = config["model"]
config_loss = config["loss"]
config_tokenizer = config["tokenizer"]
config_tokenizer["vocab_size"] = 128
print('A', sys.version)
print('B', torch.__version__)
print('C', torch.cuda.is_available())
print('D', torch.backends.cudnn.enabled)
device = torch.device('cuda')
print('E', torch.cuda.get_device_properties(device))
print('F', torch.tensor([1.0, 2.0]).cuda())
#print(config_tokenizer)
# load model
model = build_model(
    model_config=config_model, loss_config=config_loss, tokenizer=config_tokenizer
)
model = model.to('cuda')
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
])

print("Model loaded. ")

# Assuming final_png_list contains your selected image paths
dataset = PNGDataset(image_paths=paths_list, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

embeddings = evaluate_clip(model, dataloader)

np.save('chest-xray14_embeddings.npy', embeddings)