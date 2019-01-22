import os
import urllib.request
import csv
import re
#import urlparse
import requests
import mimetypes

validation_file = '/home/srallaba/Downloads/Validation_GCC-1.1.0-Validation.tsv'


def get_data(filename, set, location='./data'):

    if not os.path.exists(location):
        os.mkdir(location)
    g = open('log_' + set, 'w')
    g.close()
  

    # TODO: Check if file exists
    count = 0
    with open(filename) as tsvfile:
         reader = csv.reader(tsvfile, delimiter='\t')
         for row in reader:
            count += 1
            if count % 100 == 1:
               print("Downloaded ", count, " files")
            url = row[1]
            print(count, url)
            dst_path = location + '/coca_' + set + '_' + str(count).zfill(10) + '.jpg' 
            try:
               urllib.request.urlretrieve(url, dst_path)
            except :
               g = open('log_' + set, 'a')
               g.write("Error with the url " + str(url) + '\n')
               g.close()

get_data(validation_file, 'val', '../data')
