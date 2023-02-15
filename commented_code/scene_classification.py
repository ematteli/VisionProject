
import os
import csv
import argparse
import requests
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as tfm


def download_scene_hierarchy_file():
    print("Downloading scene hierarchy file from places365")
    # quresto file ci dice come passare dalle 365 scene classes originali alle 16 intermedie nostre e poi alle 3 finali
    response = requests.get('https://docs.google.com/spreadsheet/ccc?key=1H7ADoEIGgbF_eXh9kcJjCs5j_r3VJwke4nebhkdzksg&output=csv')
    with open("scene_hierarchy_places365.csv", "w") as file:
        file.write(response.content.decode() + "\n")

def download_pretrained_on_places(model_name="resnet50"):
    print(f"Downloading trained {model_name} on places365")
    #il modello per fare scene classification
    response = requests.get(f'http://places2.csail.mit.edu/models_places365/{model_name}_places365.pth.tar')
    with open(f"{model_name}_places365.pth.tar", "wb") as file:
        file.write(response.content)

class SceneClassifier(torch.nn.Module):

    def __init__(self, scene_hierarchy_file='scene_hierarchy_places365.csv', model_name="resnet50"):
        super().__init__()
        #da quel che dicono loro nel paper hanno usato resenet152
        assert model_name in ["resnet50", "resnet152"], f"model_name is {model_name}, should be resnet50 or resnet152"
        if not os.path.exists(scene_hierarchy_file):
            #scarica il file di cui sopra
            download_scene_hierarchy_file()
        # read scene_hierarchy file to get lvl1 meta information
        print('Loading scene hierarchy ...')
        hierarchy_places3 = []
        with open(scene_hierarchy_file, 'r') as csvfile:
            content = csv.reader(csvfile, delimiter=',')
            next(content)  # skip explanation line
            next(content)  # skip explanation line
            next(content) # skip explanation line
            for line in content:
                if len(line)!=0:
                    hierarchy_places3.append(line[1:4])
            print(hierarchy_places3[0:10])
                
                
        #trasforma il file csv in una matrice
        hierarchy_places3 = np.asarray(hierarchy_places3, dtype=float)
        
        # normalize label if it belongs to multiple categories
        self.hierarchy_places3 = hierarchy_places3 / np.expand_dims(np.sum(hierarchy_places3, axis=1), axis=-1)
        
        if not os.path.exists(f"{model_name}_places365.pth.tar"):
            #scarica il pretrained model
            download_pretrained_on_places(model_name)
        #l'attributo model dell'oggetto SceneClassifier è solo resenet50...strano
        self.model = torchvision.models.resnet50(num_classes=365)
        
        #qua bisognerebbe vedere questo modello {model_name}_places365.pth.tar e capire cosa ha in "state_dict"
        state_dict = torch.load(f"{model_name}_places365.pth.tar")["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        self.transform = tfm.Compose([
            tfm.ToTensor(),
            tfm.Resize([256, 256]),
            tfm.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    #metodo per processare le immagini nel nostro dataset
    def process_images(self, images_paths):
        pil_images = [Image.open(p) for p in images_paths]
        batch = [self.transform(p) for p in pil_images]
        batch = torch.stack(batch)
        return self(batch)
    
    def forward(self, batch):
        # Return a list of number that can be 0, 1 or 2
        with torch.inference_mode():
            b, c, h, w = batch.shape
            scene_probs = self.model(batch)
            places_prob = np.matmul(scene_probs, self.hierarchy_places3)
            scene_label_int = np.argmax(places_prob, axis=1)
        return scene_label_int.tolist()
    
    def label_int_to_str(self, scene_label_int):
        if scene_label_int == 0:
            return 'indoor'
        elif scene_label_int == 1:
            return 'natural'
        elif scene_label_int == 2:
            return 'urban'


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image_path', type=str, required=True, help='path to image file')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    scene_classifier = SceneClassifier()
    places_prob = scene_classifier.process_images([args.image_path])
    label = scene_classifier.label_int_to_str(places_prob[0])
    print(f"Image {args.image_path} has label {label}")
