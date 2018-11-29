import pickle
from Utils import text_processing

with open('imageid2captions.pkl', 'rb') as f:
   imageid2captions = pickle.load(f)

words = set()
for sample in imageid2captions:
    words.update(text_processing.tokenize(imageid2captions[sample]))

with open('imageid2captions_val.pkl', 'rb') as f:
   imageid2captions = pickle.load(f)

for sample in imageid2captions:
    words.update(text_processing.tokenize(imageid2captions[sample]))

words.update(['<start>', '<end>', '<unk>'])
words = sorted(words)
with open('vocab_captions_vqa.txt', 'w') as f:
    for item in words:
        f.write("%s\n" % item)
