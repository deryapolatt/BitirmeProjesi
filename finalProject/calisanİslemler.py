import pandas as pd
import numpy as np
import joblib

random_forest = joblib.load("./random_forest.joblib")

df = pd.read_csv("e__train.csv")
depart = np.array(df['Department'])
depart = np.unique(depart)

edField = np.array(df['EducationField'])
edField = np.unique(edField)

def employee_analysis(age,dailyRate,departmentName, edFieldName,yearsAtComp,yearsInCurrent,lastPromo):

    girdi = np.array([age,dailyRate])
    for w in depart:
        if departmentName == w:
            girdi = np.append(girdi,1)
        else:
            girdi = np.append(girdi,0)
    for w in edField:
        if edFieldName == w:
            girdi = np.append(girdi,1)
        else:
            girdi = np.append(girdi,0)

    girdi = np.append(girdi,[yearsAtComp,yearsInCurrent,lastPromo])
    girdi= girdi.reshape(1, -1)

    tahmin = random_forest.predict(girdi)
    return tahmin

def calisan_response():
    return "Yaş, Günlük oran, Departman, Eğitim alanı, Şirkette kaç yıldır çalıştığı," \
           " Şu anki pozisyonda kaç yıldır çalıştığı ve en son promotiondan kaç yıl geçtiği bilgileri kaç yıl önce alındığı bilgilerini sırasıyla girebilir misiniz? "

def hesapla_performance(sent):
    text = sent.split(", ")
    result = employee_analysis(int(text[0]),int(text[1]),text[2],text[3],int(text[4]),int(text[5]),int(text[6]))[0]
    return "Tahmin: " + result

#print(employee_analysis(32,279,'Sales','Life Sciences',6,4,0))
#print(hesapla_performance("45, 234, Sales, Life Sciences, 3, 4, 5")[0])