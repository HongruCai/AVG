import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import json


class EmbDataset(data.Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)


class CapEmbDataset(data.Dataset):
    def __init__(self,img_path,cap_path):

        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.img_embeddings = np.load(img_path)
        self.cap_embeddings = np.load(cap_path)
        self.dim = self.img_embeddings.shape[-1]

    def __getitem__(self, index):
        img_emb = self.img_embeddings[index]
        cap_emb = self.cap_embeddings[index]
        img_emb,cap_emb=torch.FloatTensor(img_emb),torch.FloatTensor(cap_emb)
        return [img_emb,cap_emb]

    def __len__(self):
        return len(self.img_embeddings)


class CapPseudoEmbDataset(data.Dataset):
    def __init__(self,img_path,cap_path,pseudo_img,pseudo_cap):

        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.img_embeddings = np.load(img_path)
        self.cap_embeddings = np.load(cap_path)
        self.pseudo_img = np.load(pseudo_img)
        self.pseudo_cap = np.load(pseudo_cap)
        self.img_embeddings = np.concatenate((self.img_embeddings,self.pseudo_img))
        self.cap_embeddings = np.concatenate((self.cap_embeddings,self.pseudo_cap))
        self.dim = self.img_embeddings.shape[-1]

    def __getitem__(self, index):
        img_emb = self.img_embeddings[index]
        cap_emb = self.cap_embeddings[index]
        img_emb,cap_emb=torch.FloatTensor(img_emb),torch.FloatTensor(cap_emb)
        return [img_emb,cap_emb]

    def __len__(self):
        return len(self.img_embeddings)


class ImgCapEmbDataset(Dataset):
    def __init__(self, root_dir, caption_file, clip_model='openai/clip-vit-large-patch14-336', device="cpu", pseudo_file=False):
        self.root_dir = root_dir
        with open(caption_file,'r') as f:
            self.captions = json.load(f)
        self.device = device
        self.images = os.listdir(root_dir)
        self.model = CLIPModel.from_pretrained(clip_model).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        self.pseudo_file = pseudo_file

    def __len__(self):
        return len(self.images) * 5

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx//5])
        image = Image.open(img_name).convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_emb = self.model.get_image_features(**inputs).detach()
            #print(img_emb.shape)
        if self.pseudo_file:
            for item in self.captions:
                if str(item['image_id']) == str(self.images[idx//5]):
                    caption = item['caption'][idx%5]
                    break
        else:
            caption = self.captions[self.images[idx//5]]['caption'][idx%5]
        
        #print(img_name,caption)
        inputs = self.processor(text=caption, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            cap_emb = self.model.get_text_features(**inputs).detach()

        return img_emb.squeeze(0), cap_emb.squeeze(0), self.images[idx//5]
    

class FilteredEmbDataset(Dataset):
    def __init__(self, data_path, indices):
        self.embeddings = np.load(data_path)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]  
        return self.embeddings[real_idx]