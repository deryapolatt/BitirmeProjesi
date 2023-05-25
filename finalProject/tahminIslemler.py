from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

modelTahmin = load_model('tahmin.h5')
df = pd.read_csv("train.csv")


df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y').dt.strftime('%Y')

sales_by_year = df.groupby('Ship Date')['Sales'].sum()

# SALES SÜTUNU İÇİN IQR DEĞERİ HESAPLAMA ARDINDAN OUTLİER DEĞERLERİN SİLİNMESİ
Q1=df.Sales.quantile(0.25)
Q3=df.Sales.quantile(0.75)
IQR=Q3-Q1
alt_limit= Q1- 1.5*IQR
üst_limit= Q3+ 1.5*IQR
df=df[(df.Sales > alt_limit) &(df.Sales <üst_limit)]


#Numeric ID Sütununun Oluşturulması
df['Numeric ID'] = df['Product ID'].str.extract('(\d+)')
grouped = df.groupby('Numeric ID')['Product ID'].nunique()
duplicated_ids = grouped[grouped > 1].index.tolist()
df = df[~df['Numeric ID'].isin(duplicated_ids)]

#İlk 500 Customer'a Göre Veri Setinin Ayarlanması
grouped_df = df.groupby("Customer Name").sum().sort_values("Sales", ascending=False)
top_500_customers = grouped_df.index[:500]
df1= df[df["Customer Name"].isin(top_500_customers)]
df=df1




#City Sütununa One-Hot Kodlama
one_hot_encoded = pd.get_dummies(df['City'])
df = pd.concat([df, one_hot_encoded], axis=1)
#Customer Name Sütununa One-Hot Kodlama

one_hot_encoded = pd.get_dummies(df['Customer Name'])
df = pd.concat([df, one_hot_encoded], axis=1)

#Kullanılmayacak Sütunların Silinmesi İşlemi

df = df.drop(['Row ID','Order Date', 'Order ID','Ship Mode','Customer ID','City','Customer Name','Segment','Country','Postal Code','Region','State','Product ID','Product Name','Category','Sub-Category','Product Name'], axis=1)

'''
# Ship Date ve Sales Normalizasyon
from sklearn.preprocessing import MinMaxScaler
columns_to_normalize = ['Ship Date', 'Sales','Numeric ID']

# Min-Max Normalizasyonu uygulama
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
df.head()
'''


all_columns = df.columns.tolist()
columns_to_normalize = all_columns
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
df.head()

def tahmin_yap(shipDate, numericId, city, customerName):

    # Tahmin İşlemi
    yeni_veri = pd.DataFrame({
        'Ship Date': [shipDate],
        'Numeric ID': [numericId],
        city: [1],  # one-hot kodlanmış city değişkenlerinden
        customerName: [1]  # one-hot kodlanmış customer name değişkenlerinden
    })
    cikarilacak_sutun = ['Sales']
    tum_sutunlar = [sutun for sutun in df.columns if sutun not in cikarilacak_sutun]

    eksik_sutunlar = set(tum_sutunlar) - set(yeni_veri.columns)
    for sutun in eksik_sutunlar:
        yeni_veri[sutun] = 0

    columns_to_normalize = tum_sutunlar
    scaler = MinMaxScaler()
    yeni_veri[columns_to_normalize] = scaler.fit_transform(yeni_veri[columns_to_normalize])
    print(yeni_veri.shape)
    tahmin = modelTahmin.predict(yeni_veri)
    return  tahmin

def before_tahmin(sent):
    text = sent.split(",")
    result = tahmin_yap(int(text[0]), int(text[1]), text[2], text[3])
    return str(result[0])
def tahmin_soru():
    return "Satış tahmin oranını öğrenmek için aşağıdaki verileri sırasıyla girmeniz gerekmektedir:" \
           " sevkiyat tarihi, numerik id, şehir ismi ve müşteri ismi"
#tahmin_yap('35','500','Los Angeles','Ken Dana')
#before_tahmin("35,500,Los Angeles,Ken Dana")
#2017,10002892,Henderson,Claire Gute