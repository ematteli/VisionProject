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