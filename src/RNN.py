import xlsxwriter
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import classification_report


def read_file_pos_list(path):
    tags = []
    file = pd.read_excel(path)
    for line in file:
        for i in range(0, 42):
            tags.append(file[line][i])
    return tags


def read_file(path):
    sentence, tag_sentence = [], []
    file = pd.read_excel(path)
    sen_col = file['Sentence #']
    word_col = file['Word']
    pos_col = file['POS']
    lengths = []
    sub = "Sentence"
    sen1 = []
    pos = []
    k = 0
    for i in file.index:
        if (not (str(sen_col[i]).find(sub))):
            k = k + 1
            if (len(sen1) == 0):
                sen1.append(word_col[i])
                pos.append(pos_col[i])
            else:
                lengths.append(len(sen1))
                sentence.append(deepcopy(sen1))
                tag_sentence.append(deepcopy(pos))
                sen1.clear()
                pos.clear()
                sen1.append(word_col[i])
                pos.append(pos_col[i])
        else:
            sen1.append(word_col[i])
            pos.append(pos_col[i])
    return sentence, tag_sentence, lengths


def predicted_fun(file_path) -> list:
    data = pd.read_excel(file_path)
    _word = data["predicted"].tolist()
    return _word




POS = read_file_pos_list('pos_list.xlsx')
train_sentences, train_tags, train_length = read_file('Rnn_train.xlsx')
test_sentences, test_tags, test_length = read_file('Rnn_test.xlsx')

real = []
for s in test_tags:
    for w in s:
        real.append(w)

# assign to each word and tag a unique integer
words = set([])
for s in train_sentences:
    for w in s:
        words.add(str(w).lower())
word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0
word2index['-OOV-'] = 1

tag2index = {t: i + 1 for i, t in enumerate(list(POS))}
tag2index['-PAD-'] = 0
train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

for s in train_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[str(w).lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    train_sentences_X.append(s_int)

for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[str(w).lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    test_sentences_X.append(s_int)

for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])

for s in test_tags:
    test_tags_y.append([tag2index[t] for t in s])

MAX_LENGTH_Train = len(max(test_sentences_X, key=len))
MAX_LENGTH_Test = len(max(test_sentences_X, key=len))

# padding
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH_Train, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH_Test, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH_Train, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH_Test, padding='post')


# evaluation metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def accuracy(to_ignore=0):
    def acc(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return acc


# network
my_model = Sequential()
my_model.add(InputLayer(input_shape=(MAX_LENGTH_Train,)))
my_model.add(Embedding(len(word2index), 128))
my_model.add(Bidirectional(LSTM(256, return_sequences=True)))
my_model.add(TimeDistributed(Dense(len(tag2index))))
my_model.add(Activation('softmax'))

my_model.compile(loss='categorical_crossentropy',
                 optimizer=Adam(0.001),
                 metrics=[precision_m, recall_m, accuracy(0)])

my_model.summary()

cat_train_tags_y = (train_tags_y, len(tag2index))
trained_model = my_model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=512, epochs=20,
                             validation_split=0.2)
scores = my_model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))

test_predict = my_model.predict(test_sentences_X)
workbook = xlsxwriter.Workbook('test.xlsx')
worksheet = workbook.add_worksheet()
row = 1
column = 5
predict = []
i = 0
for s in test_predict:
    x = test_length[i]
    for w in s[0:x]:
        predicted = POS[np.argmax(w)]
        worksheet.write(row, column, predicted)
        predict.append(predicted)
        row += 1
    i += 1
workbook.close()


_Pos = predicted_fun('Rnn_test.xlsx')
del _Pos[-1]
print(_Pos)



classification_reports = classification_report(real, _Pos)
print(classification_reports)

plt.title('loss')
plt.plot(trained_model.history['loss'], label='train')
plt.plot(trained_model.history['val_loss'], label='validation')
plt.legend()
plt.show()
# plot accuracy during training
plt.title('Accuracy')
plt.plot(trained_model.history['acc'], label='train')
plt.plot(trained_model.history['val_acc'], label='validation')
plt.legend()
plt.show()
# plot precision during training
plt.title('precision')
plt.plot(trained_model.history['precision_m'], label='train')
plt.plot(trained_model.history['val_precision_m'], label='validation')
plt.legend()
plt.show()

# plot recall during training
plt.title('recall')
plt.plot(trained_model.history['recall_m'], label='train')
plt.plot(trained_model.history['val_recall_m'], label='validation')
plt.legend()
plt.show()
