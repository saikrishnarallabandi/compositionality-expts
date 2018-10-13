# This script just extracts questions from CVQA json files
import os
import json

cvqa_directory = '/data/CVQA'

files = sorted(os.listdir(cvqa_directory))

for file in files:
   if file.endswith('.json') and 'questions' in file:
      print("Extracting questions from ", file)
      g = open(file.split('.json')[0] + '.txt','w')
      with open(cvqa_directory + '/' + file) as json_file:
          json_data = json.load(json_file)
          for q in json_data:
              g.write(q['question'] + '\n')
      g.close()
