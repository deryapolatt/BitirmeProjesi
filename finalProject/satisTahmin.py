import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras import regularizers



df = pd.read_csv("train.csv")

cities = np.array(df['City'])
cities = np.unique(cities)

customerNames = np.array(df['Customer Name'])
customerNames = np.unique(customerNames)


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
one_hot_encoded
df = pd.concat([df, one_hot_encoded], axis=1)
#Customer Name Sütununa One-Hot Kodlama

one_hot_encoded = pd.get_dummies(df['Customer Name'])
one_hot_encoded
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
print(df.head())




# 4 KATMANLI DNN MODELİ


X = df.drop(['Sales'], axis=1).values
y = df['Sales'].values

train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size, :], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size, :], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:, :], y[train_size+val_size:]

model = Sequential()

model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=20, batch_size=24, validation_data=(X_test, y_test))
model.save('tahmin.h5',history)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('4 KATMANLI D')
plt.legend()
plt.show()
