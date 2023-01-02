
import nltk
import pandas as pd
import xlsxwriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def reshape_data(file_path) -> list:
    data = pd.read_excel(file_path)
    _sentence_num = data["Sentence #"].tolist()
    _word = data["Word"].tolist()
    _pos = data["POS"].tolist()

    re_data = list()  # reshaped data
    words = list()
    pos = list()
    ci = -1
    for i in range(len(_sentence_num)):
        if type(_sentence_num[i]) == str:
            ci += 1
            re_data.append(list())
            words.append(list())
            pos.append(list())

        re_data[ci].append((_word[i], _pos[i]))
        words[ci].append(_word[i])
        pos[ci].append(_pos[i])

    sent_tag = []
    for s in re_data:
        s.insert(0, ('<s>', '<s>'))
        s.append(('</s>', '</s>'))
        sent_tag.append(s)

    test_words = list()
    test_tags = list()
    for s in sent_tag:
        sen_word = []
        sen_pos = []
        for (w, p) in s:
            sen_word.append(str(w).lower())
            sen_pos.append(p)
        test_words.append(sen_word)
        test_tags.append(sen_pos)

    return sent_tag, test_words, test_tags


def read_file_pos_list(path):
    tags = []
    file = pd.read_excel(path)
    for line in file:
        for i in range(0, 42):
            tags.append(file[line][i])
    return tags


def word_pos_count():
    pos = {}
    count = 0
    for sen in train_data:
        for (w, t) in sen:
            w = str(w).lower()
            try:
                if t not in pos[w]:
                    pos[w].append(t)
            except:
                new_element = list()
                new_element.append(t)
                pos[w] = new_element

    for s in test_data:
        for (w, t) in s:
            w = str(w).lower()
            try:
                if t not in pos[w]:
                    pos[w].append(t)
            except:
                new_element = list()
                new_element.append(t)
                pos[w] = new_element
    return pos


def create_key_value(data):
    dictionary_word_tag = {}
    for s in data:
        for (w, t) in s:
            w = str(w).lower()
            try:
                try:
                    dictionary_word_tag[t][w] += 1
                except:
                    dictionary_word_tag[t][w] = 1
            except:
                dictionary_word_tag[t] = {w: 1}

    return dictionary_word_tag


def emission(dictionary):
    emission = {}
    for i in dictionary.keys():
        emission[i] = {}
        summation = sum(dictionary[i].values())
        for j in dictionary[i].keys():
            emission[i][j] = dictionary[i][j] / summation
            if (emission[i][j] == 0):
                print("zero")
    return emission


def bigram_POS(data):
    bigram_POS_data = {}
    for s in data:
        bigram = list(nltk.bigrams(s))
        for bigram1, bigram2 in bigram:
            try:
                try:
                    bigram_POS_data[bigram1[1]][bigram2[1]] += 1
                except:
                    bigram_POS_data[bigram1[1]][bigram2[1]] = 1
            except:
                bigram_POS_data[bigram1[1]] = {bigram2[1]: 1}
    return bigram_POS_data

def transition(data):
    bigram_POS_prob = {}
    for i in data.keys():
        bigram_POS_prob[i] = {}
        summation = sum(data[i].values())
        for j in data[i].keys():
            bigram_POS_prob[i][j] = data[i][j] / summation
            if (bigram_POS_prob[i][j] == 0):
                print("zero")
    return bigram_POS_prob

def viterbi():
    predicted = []
    for sen in range(len(test_words)):
        sentence = test_words[sen]
        _veterbi = {}
        for word in range(len(sentence)):
            step = sentence[word]
            # for the starting word of the sentence
            if word == 1:
                _veterbi[word] = {}
                tags = pos[step]
                for t in tags:
                    # if is first word of a sentence
                    try:
                        _veterbi[word][t] = ['<s>', bigram_POS_prob['<s>'][t] * train_emission_prob[t][step]]
                    # if it is not the first word of a sentence
                    except:
                        _veterbi[word][t] = ['<s>', 0.00001]

            # if it  is not first word of a sentence
            if word > 1:
                _veterbi[word] = {}
                previous = list(_veterbi[word - 1].keys())
                current= pos[step]
                for t in current:
                    temp = []
                    for pt in previous:
                        try:
                            temp.append(
                                _veterbi[word - 1][pt][1] * bigram_POS_prob[pt][t] * train_emission_prob[t][step])
                        except:
                            temp.append(_veterbi[word - 1][pt][1] * 0.00001)
                    max_temp_index = temp.index(max(temp))
                    best_pt = previous[max_temp_index]
                    _veterbi[word][t] = [best_pt, max(temp)]

        # Backtrack
        pred_tags = []
        total_steps = _veterbi.keys()
        last_step = max(total_steps)
        for bs in range(len(total_steps)):
            step_num = last_step- bs
            if step_num == last_step:
                pred_tags.append('</s>')
                pred_tags.append(_veterbi[step_num]['</s>'][0])
            if step_num < last_step and step_num > 0:
                pred_tags.append(_veterbi[step_num][pred_tags[len(pred_tags) - 1]][0])
        predicted.append(list(reversed(pred_tags)))
    return predicted


POS = read_file_pos_list('pos_list.xlsx')
train_data, train_words, train_pos = reshape_data('viterbi_train.xlsx')
test_data, test_words, test_tags = reshape_data('viterbi_test.xlsx')
pos = word_pos_count()
dic_word_tag = create_key_value(train_data)
train_emission_prob = emission(dic_word_tag)
bigram_POS_data = bigram_POS(train_data)
bigram_POS_prob=transition(bigram_POS_data)
predicted=viterbi()

# metrics
row = 1
column = 5
y_pred = list()
y_true = list()
workbook = xlsxwriter.Workbook('test1.xlsx')
worksheet = workbook.add_worksheet()
for sen in predicted:
    for tag in sen:
        if (tag not in ('<s>', '</s>')):
            worksheet.write(row, column, tag)
            y_pred.append(tag)
            # print(tag)
            row += 1
for sen in test_tags:
    for tag in sen:
        if (tag not in ('<s>', '</s>')):
            y_true.append(tag)
            # print(tag)
            row += 1
workbook.close()
print(POS)
uni = set()
for m in y_pred:
    uni.add(m)
print(uni)
print(len(uni))
for k in POS:
    if (k not in uni):
        print(k)
del POS[-1]
classification_reports = classification_report(y_true, y_pred, target_names=POS)
print(classification_reports)
print("accuracy:", accuracy_score(y_true, y_pred))

