import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, batch_size):
        self.dictionary = Dictionary()
        self.PAD_IDX = self.dictionary.add_word('PAD')
        #print "init"
        self.train, self.train_type = self.tokenize(os.path.join(path, 'train2014.questions.txt'), batch_size)
        self.valid, self.valid_type = self.tokenize(os.path.join(path, 'val2014.questions.txt'), batch_size)
        self.test, self.test_type = self.tokenize(os.path.join(path, 'val2014.questions.txt'), 1)
        # Tokenize, pad and batchify and UNK words

        #return self.train, self.valid, self.test



    # add words in the dictionary

    def tokenize(self, path, batch_size):
        """Tokenizes a text file."""
        #print path
        #print "tokenize"
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = ['<sos>'] + line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    if word[-1]=="?":
                        word = word[:len(word)-1]
                        self.dictionary.add_word("?")
                    self.dictionary.add_word(word)

        # Tokenize file content

        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            all_samples = []
            for line in f:
                words = ['<sos>']+ line.split() + ['<eos>']
                tokens = []
                for word in words:
                    found_question = False
                    if word[-1]=="?":
                        word = word[:len(word)-1]
                        found_question = True
                    token = self.dictionary.word2idx[word]
                    tokens.append(token)
                    if found_question:
                        token = self.dictionary.word2idx["?"]
                        found_question = False
                        tokens.append(token)
                    #tokens.append(token)
                all_samples.append(tokens)
            return self.batchify(all_samples, batch_size)

    def batchify(self, all_samples, batch_size):
        #print "batchify"
        #print "batchify"
        batched_samples = [] # each entry is a batch X seq_length tensor
        batched_types = []
        all_samples.sort(key = lambda s: len(s))
        b = 0
        while(b<len(all_samples)):
            #print b, b+batch_size
            batch = all_samples[b:b+batch_size]
            # batch in all_samples pad it
            temp_batch = []
            temp_type = []
            max_length = len(batch[-1])
            for sample in batch:

                # sample is a list

                if sample[1]==self.dictionary.word2idx["Is"] or sample[1]==self.dictionary.word2idx["Are"]:
                    type_1 = [0] # yes/no type
                    #print ("Is")
                elif  sample[1]==self.dictionary.word2idx["How"] and sample[2]==self.dictionary.word2idx["many"]:
                    type_1 = [1] # count
                    #print ("Count")
                else:
                    type_1 = [2] # other

                #print(sample[0], self.dictionary.idx2word[sample[0]])
                while (len(sample)<max_length):
                    sample.append(self.PAD_IDX) # pad each example
                #print len(sample)
                temp_batch.append(torch.LongTensor(sample))
                temp_type.append(torch.LongTensor(type_1))
            # stack into a tensor
            b += batch_size
            batched_samples.append(torch.stack(temp_batch, dim=0))
            batched_types.append(torch.stack(temp_type, dim=0))
            #all_samples.torch.stack(temp_batch, 0)
        #exit(1)
        #for a in batched_samples:
        #    print a.size()
        return batched_samples, batched_types
