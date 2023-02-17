import pandas as pd
import requests
import numpy as np
import time
import os

#I took this dataset from https://www.kaggle.com/datasets/amiralisa/flickr/download?datasetVersionNumber=1
csv10k = r'C:\Users\latta\GitHub\Vision_Project\GeoEstimation\resources\images\landscape10k\data_10k.csv'
new_data_urls = pd.read_csv(csv10k, sep = ';')
print(new_data_urls.head(2))

'''
This is the way to find the url from our dataframe (info from https://www.flickr.com/services/api/misc.urls.html):
 - sometimes https://www.flickr.com/photos/{user-id}/{photo-id} works sometimes no
 - we can also try https://www.flickr.com/photo.gne?id={photo-id}
'''

#add the new column with the base_url
ss = 'https://www.flickr.com/photos/'
new_data_urls['base_url']=[ss + '/'.join((str(i[0]),str(i[1]))) for i in zip(new_data_urls['owner'], new_data_urls['photo_id'])]
print(new_data_urls.head(2))

'''
to download the images from flickr using an url, without a webscraper we need to pass
from www.flickr.com - like urls to live.staticflickr.com ones:
- our url are like  https://www.flickr.com/photos/130418712@N05/17271526139
- we need something like https://live.staticflickr.com/3451/3228511372_33d4273d03_o.jpg
We can do this using the package request: https://requests.readthedocs.io/en/latest/
'''

cwd = os.getcwd()
error_404 = 0
url_list = {}
exception = {}
count = 1
for i,url in enumerate(new_data_urls['base_url']):
    #to account for some connection error we use a 'try'
    try:
        r = requests.get(url)
        index = r.text.find('https://live.staticflickr.com')
        if index == -1:
            error_404 += 1
            url_list[new_data_urls['photo_id'][i]]=np.nan
        else:
            new_url = r.text[index:index+64]
            if new_url.find('jpg')==-1:
                print('this url is not a .jpg')
                url_list[new_data_urls['photo_id'][i]]=np.nan
            else:
                new_url = new_url[:new_url.find('jpg')+3]
                url_list[new_data_urls['photo_id'][i]]=new_url
    except:
        time.sleep(30)
        exception[new_data_urls['photo_id'][i]]=url
        print("exception! ", count)
        count += 1
 
    if (i+1)%1000==0:
        print(f"we are at the {i}-th url")

        
print('error_404: ',error_404) #more or less 500 urls do not work any more
print(exception)

#let's save our new dataframes
partial_urls = pd.DataFrame(data = url_list.values(), index = url_list.keys(), columns=['photo_id','url'])
partial_urls.to_csv(path_or_buf=cwd + r'\resources\images\new_data10k\data_10k_with_urls.csv', sep=';')
dataset = pd.read_csv(cwd + r'\resources\images\new_data10k\data_10k_with_urls.csv', sep=';')
dataset.to_csv(path_or_buf=cwd + r'\resources\images\new_data10k\data_10k_with_urls.csv', sep=';')
final_dataset = new_data_urls.merge(right  = dataset, on='photo_id')
final_dataset.to_csv(path_or_buf=r'C:\Users\latta\GitHub\Vision_Project\GeoEstimation\resources\images\new_data10k\final_dataset.csv', sep=';')
print(final_dataset.head())

import os
import sys
import urllib.request
#our defult working directory is Geoestimation

def download_image(url, file_path, file_name, size = 'z'):
    url = url[:-5]+size+url[-4:]
    file_name = str(file_name)
    full_path = file_path + '/'+ file_name + '.jpg'
    try: 
        urllib.request.urlretrieve(url, full_path)
        return 'ok'
    except:
        print(f'the url {url} does not work')
        return ''


def download_from_dataframe(df, num_photos=1500):
    os.chdir(r'/content/drive/MyDrive/GeoEstimation/resources/images/new_data10k')
    cwd = os.getcwd()
    count = 0
    start = os.listdir()
    if len(start)>10006:
        print('Dataset already downloaded')
        return
    for i,url in enumerate(df['url']):
        id = str(df['photo_id'][i])
        
        if id+'.jpg' not in start and count<num_photos and type(url)!=float:
            status = download_image(url, cwd , id)
            if status=='ok':
                count += 1
    #this is to return to the original parent folder
    os.chdir(r'..')
    os.chdir(r'..')
    os.chdir(r'..')
    return 

import pandas as pd

new_data = pd.read_csv(r'/content/drive/MyDrive/GeoEstimation/resources/images/final_dataset_10k.csv', sep = ';', index_col = 0)
new_data.head(2)
print('Previous number of Photos:',len(os.listdir(r'/content/drive/MyDrive/GeoEstimation/resources/images/new_data10k'))-1)
download_from_dataframe(new_data)
print('New number of Photos:',len(os.listdir(r'/content/drive/MyDrive/GeoEstimation/resources/images/new_data10k'))-1)

#FINAL OUTPUT
'''Previous number of Photos: 9447
the url https://live.staticflickr.com/7878/40415896353_960ee7376z.jpg does not work
New number of Photos: 9447'''

#at this point we need the labels for this images as urban, indoor or natural with relative probabilites
#we need their classifier

import numpy as np
import importlib
imported_module = importlib.import_module("scene_classification")
importlib.reload(imported_module)
import scene_classification
from scene_classification import SceneClassifier

import pandas as pd

#initialiaze the classifier
scene_classifier = SceneClassifier(runtime='cpu')

#list of the images with full path
path = r'/content/drive/MyDrive/GeoEstimation/resources/images/new_data10k'
path_list = os.listdir(path)
path_list = [path+'/'+im for im in path_list if im[-3:]=='jpg']
print('num of images is ' , len(path_list))

#original file csv with images info
data_10k = pd.read_csv(r'/content/drive/MyDrive/GeoEstimation/resources/images/final_dataset_10k.csv', sep=';')
print('the origninal df of images info is')
print(data_10k.head())

#classification of the images, producing both S3_labels that probs
places_prob, S3_labels = scene_classifier.process_images(path_list,b_size=128)
print('num of triplette of probabilities is ' , len(places_prob))
print('num of  S3_labels is ', len(S3_labels))

#new file csv formation
new_data = pd.DataFrame(data = np.asarray(places_prob), columns=['Prob_indoor','Prob_natural','Prob_urban'])
S3_labels_data =  pd.DataFrame(data = np.asarray([S3_labels]).T ,columns=['S3_label'])
images_name = [im[:-4] for im in os.listdir(path) if im[-3:]=='jpg']
images_name_df = pd.DataFrame(data = images_name ,  columns=['photo_id'])

#new df with the new information
new_df = pd.concat([images_name_df,S3_labels_data,new_data],axis=1)
print('the new data column we have now are')
print(new_df.head(20))

#final merge with the original csv
new_df_full = data_10k.merge(right = new_df, on='photo_id')
print('final DataFrame')
new_df_full.head()

#save the new dataframe in a csv
print("Let's save the results")
new_df_full.to_csv(path_or_buf=r'/content/drive/MyDrive/GeoEstimation/resources/images/data10k_places365.csv', sep=',')