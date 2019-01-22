

# two references for one document
from nltk.translate.bleu_score import *
references = [['<start>', 'three', 'little', 'girls', 'posing', 'in', 'front', 'of', 'a', 'wall', 'for', 'a', 'photo', '.', '<end>']] 
candidates = ['<start>', 'a', 'young', 'man', 'is', 'playing', 'tennis', 'on', 'a', 'court', '.', '<end>']
chencherry = SmoothingFunction()
score = sentence_bleu(references, candidates, smoothing_function=chencherry.method1)
print(score)
