import numpy as np
import random
import json


def process_snli(file_path, word_to_index, to_lower):
    label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            example = {}
            line = json.loads(line)
            if line['gold_label'] != '-':
                example['label'] = label_dict[line['gold_label']]
                if to_lower is True:
                    tmp1 = line['sentence1_binary_parse'].replace('(', '').replace(')', '').lower().split()
                    tmp2 = line['sentence2_binary_parse'].replace('(', '').replace(')', '').lower().split()
                else:
                    tmp1 = line['sentence1_binary_parse'].replace('(', '').replace(')', '').split()
                    tmp2 = line['sentence2_binary_parse'].replace('(', '').replace(')', '').split()
                tmp1.insert(0, '<NULL>')
                tmp2.insert(0, '<NULL>') 
                example['premise'] = ' '.join(tmp1) 
                example['hypothesis'] = ' '.join(tmp2)
                example['premise_to_words'] = [word for word in example['premise'].split(' ')]
                example['hypothesis_to_words'] = [word for word in example['hypothesis'].split(' ')]
                example['premise_to_tokens'] = [word_to_index[word] if word in word_to_index.keys() else (hash(word) % 100) for word in example['premise_to_words']]
                example['hypothesis_to_tokens'] = [word_to_index[word] if word in word_to_index.keys() else (hash(word) % 100) for word in example['hypothesis_to_words']]
                example['max_length'] = max(len(example['premise_to_tokens']), len(example['hypothesis_to_tokens']))
                data.append(example)
    return data


def load_embedding_and_build_vocab(file_path):
    vocab = []
    word_embeddings = []
    for i in range(0, 100): 
        oov_word = '<OOV' + str(i) + '>'
        vocab.append(oov_word)
        word_embeddings.append(list(np.random.normal(scale=1, size=300)))
    vocab.append('<PAD>')
    word_embeddings.append(list(np.zeros(300)))
    index_to_word = dict(enumerate(vocab))
    word_to_index = dict([(index_to_word[index], index) for index in index_to_word])

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.split()
            word = ' '.join(line[:-300])
            vector = [float(x) for x in line[-300:]]
            vocab.append(word)
            word_embeddings.append(vector)
            word_to_index[word] = i + 101
            index_to_word[i + 101] = word

    return vocab, np.array(word_embeddings), word_to_index, index_to_word


def semi_sort_data(data):
    return [example for example in data if example['max_length'] < 20] + \
           [example for example in data if 20 <= example['max_length'] < 50] +\
           [example for example in data if example['max_length'] >= 50]


def batch_iter(dataset, batch_size, shuffle):
    start = -1 * batch_size
    dataset_size = len(dataset)

    if shuffle:
        semi_sort_data(dataset)
    
    index_list = list(range(len(dataset)))

    while True:
        start += batch_size
        label = []
        premise = []
        hypothesis = []
        if start > dataset_size - batch_size:
            start = 0
        batch_indices = index_list[start:start + batch_size]
        batch = [dataset[index] for index in batch_indices]
        for k in batch:
            label.append(k['label'])
            premise.append(k['premise_to_tokens'])
            hypothesis.append(k['hypothesis_to_tokens'])
        max_length_prem = max([len(item) for item in premise])
        max_length_hypo = max([len(item) for item in hypothesis])
        for item in premise:
            item.extend([100] * (max_length_prem - len(item)))
        for item in hypothesis:
            item.extend([100] * (max_length_hypo - len(item)))
        yield [label, premise, hypothesis] 


if __name__ == '__main__':
    # Test
    vocab, word_embeddings, word_to_index, index_to_word = load_embedding_and_build_vocab('../data/glove.6B.300d.txt')
    dev_set = process_snli('../data/snli_1.0_dev.jsonl', word_to_index, to_lower=True)
    dev_iter = batch_iter(dataset=dev_set, batch_size=4, shuffle=True)
