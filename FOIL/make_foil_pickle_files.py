import json
import pickle
import sys, os

# loading the train foil
with open('./foilv1.0_train_2017.json', 'r') as f:
    array = json.load(f)

# loading the train image2idfeatures
with open('./imageid2features.pkl', 'rb') as f:
    imageid2features = pickle.load(f)

train_samples = []
annotations = array['annotations']
for item in annotations:
    image_str = 'COCO_train2014_000000'+str(item['image_id']).zfill(6)+'.jpg'
    if image_str not in imageid2features.keys():
        print(image_str)
        print("The above sample is missing")
        sys.exit(1)

    item['image_name']=image_str
    item['feature']=imageid2features[image_str]

    train_samples.append(item)

with open('./foil_train_with_features.pkl', 'wb') as f:
    pickle.dump(train_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

# loading the train foil
with open('./foilv1.0_test_2017.json', 'r') as f:
    array = json.load(f)

# loading the train image2idfeatures
with open('./imageid2features_val.pkl', 'rb') as f:
    imageid2features = pickle.load(f)

val_samples = []
annotations = array['annotations']
for item in annotations:
    image_str = 'COCO_val2014_000000'+str(item['image_id']).zfill(6)+'.jpg'
    if image_str not in imageid2features.keys():
        print(image_str)
        print("The above sample is missing")
        sys.exit(1)

    item['image_name']=image_str
    item['feature']=imageid2features[image_str]

    val_samples.append(item)

with open('./foil_val_with_features.pkl', 'wb') as f:
    pickle.dump(val_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

