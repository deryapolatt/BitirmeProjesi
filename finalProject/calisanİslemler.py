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
           " Şu anki pozisyonda kaç yıldır çalıştığı ve en son promotiondan kaç yıl geçtiği bilgileri sırasıyla girebilir misiniz? "

def hesapla_performance(sent):
    text = sent.split(",")
    result = employee_analysis(int(text[0]), int(text[1]), text[2], text[3], int(text[4]), int(text[5]), int(text[6]))[
        0]
    if result == "Yes":
        return "Yapılan tahmine göre yaşı: " + text[0] + ", günlük oran değeri: " + text[1] + ", departmanı: " + text[2] \
               + ", eğitim alanı: " + text[3] + ", şirkette " + text[4] + " yıldır çalışan, " + "şu anki pozisyonunda " \
               + text[5] + " yıldır çalışan ve " + "en son promotiondan " + text[
                   6] + " yıl geçen çalışanın yıpranma oranı yüksektir"

    elif result == "No":
        return "Yapılan tahmine göre yaşı: " + text[0] + ", günlük oran değeri: " + text[1] + ", departmanı: " + text[2] \
               + ", eğitim alanı: " + text[3] + ", şirkette " + text[4] + " yıldır çalışan, " + "şu anki pozisyonunda " \
               + text[5] + "yıldır çalışan ve " + "en son promotiondan " + text[
                   6] + " yıl geçen çalışanın yıpranma oranı düşüktür"


#print(employee_analysis(32,279,'Sales','Life Sciences',6,4,0))
#print(hesapla_performance("43, 1273, Research & Development, Medical, 3, 4, 5"))
#27,1392,Sales,Other,3,4,5
