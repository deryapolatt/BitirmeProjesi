import numpy as np
from trnlp import *
from trnlp import TrnlpWord
import json
import random
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation,Dropout,LSTM
from keras.models import Sequential, load_model
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import tensorflow as tf
#VERI SETI
intents = json.loads(open("intents.json", encoding="utf8").read())
obj = TrnlpWord()

words = []
classes = [] #tags
documents = [] # etiketli kelime listesi
ignore_letters = ["!","?",".",","]

#veri seti temizlenir/ayrıştırılır
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = simple_token(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#kök ve eklere ayrılır
newWords = []
for word in words:
  if word not in ignore_letters:
    obj.setword(word.lower())
    newWords.append(obj.get_stem)

#kelimelerin kopyasını silmek ve sıralamak
newWords = sorted(set(newWords))
classes = sorted(tuple(classes))

#Pickling, bir python nesnesini sabit sürücünüzde ikili dosya olarak kaydetmenizi sağlar
pickle.dump(newWords, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# şimdi SİNİR AĞINI train etmek için bu KELİMELERİ SAYISAL DEĞERLERE dönüştürmemiz gerekiyor
training = []
output_empty =  [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    wordPatterns = []
    for word in word_patterns:
        obj.setword(word.lower())
        wordPatterns.append(obj.get_stem)
    word_patterns = wordPatterns

    for word in newWords:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append((bag, output_row))

random.shuffle(training)
training = np.array(training,dtype=object)

#özelliklerin değerlerini içerir
train_x = list(training[:, 0])

#eğitim sürecinden sonra hangi değerleri bulmalıyız
train_y = list(training[:, 1])

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

#EĞİTİM/KATMANLAR
model = Sequential()
model.add(Dense(256,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(train_y[0]),activation='softmax'))

#Optimizer - yerine 'adam' de kullanılabilir
sgd = SGD(learning_rate=1e-1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

hist = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=30, verbose=1, validation_data=(X_test,y_test),callbacks=[early_stopping])

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.show()

model.summary()
model.save('chatbotmodel.h5',hist)
print('Done')
