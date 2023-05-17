#Kütüphaneler
from keras.models import load_model
import json
import pickle
import nltk
from trnlp import TrnlpWord
import numpy as np
import random
from graphics import top_musteri,top_grossing_cities,getPredict,urun_kategori,toplam_satis,calisan_dagilimi,\
    actual_productivity_rate,Attrcalisan_yas_dagilim,calisan_yas_dagilim, attrition_yas_aralik
from calisanİslemler import hesapla_performance,calisan_response
from trnlp import SpellingCorrector


#ILK KEZ ÇALIŞTIRIYORSAN UNCOMMENT YAP
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

intents = json.loads(open("intents.json", encoding="utf8").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')
ignore_letters = ["!","?",".",","]
obj = TrnlpWord()


#Cümleyi temizleme işlemi
def clean_up_sentence(sentence):
    result = []
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [w for w in sentence_words  if w not in ignore_letters]
    for word in sentence_words:
      obj.setword(word.lower())
      result.append(obj.get_stem)
    print("Result: ",result)
    return result

#Cümleyi BOW - Bag Of Words - 0 ve 1 lere dönüştürme
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words) # 00000
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#PREDICTION - tahmin
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r> ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'tag': classes[r[0]], 'probability':str(r[1])})
    return return_list

#gramer, yazım denetleyicisi
def correctSpelling(message):
  sentence = ""
  obj = SpellingCorrector()
  obj.settext(message)
  for i in obj.correction(all=True):
    sentence+=i[0] + " "
  return sentence

#satış tahmini için İSİM, ORG, LOC, TARİH vs. tespit etmek için
def ner(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    result = {'nnp':''}
    #print(preprocess("2024 yılında Black Box ürününün satış tahmini"))
    for i in sent:
      if 'CD' in i:
        result['cd'] = i[0]
      if 'NNP' in i:
        temp = result['nnp']
        temp+=i[0] + " "
        result['nnp'] = temp
    return result

def get_response(intents_list, intents_json, message):
    tag = intents_list[0]['tag']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
              result = random.choice(i['responses'])
              if "_" in result:
                  if result == "hesapla_performance":
                      return hesapla_performance(message)
                  else:
                      return  globals()[result]()
              if i['tag'] == 'tahmin':
                res = ner(message)
                res = getPredict(res['cd'],res['nnp'])
                result = res

    return result

#SADECE CONSOLE DA ÇALIŞTIRMAK İSTENİRSE UNCOMMENT YAP
"""print("GO! Bot çalışmaya başladı!")
while True:
    print("Ben: ")
    message = input("")
    correctedSentence = correctSpelling(message)
    ints = predict_class(correctedSentence)
    print(ints)
    res = get_response(ints, intents, message)
    print("Chatbot: ")
    print(res)
"""