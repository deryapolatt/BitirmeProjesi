import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("train.csv")
df2 = pd.read_csv("train_dataset.csv")
calisan = pd.read_csv("e__train.csv")

def toplam_satis():
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y').dt.strftime('%Y')

    sales_by_year = df.groupby('Ship Date')['Sales'].sum()

    plt.bar(sales_by_year.index, sales_by_year.values)
    plt.title('Yıllara Göre Toplam Satış')
    plt.xlabel('Yıl')
    plt.ylabel('Toplam Satış')
    plt.savefig("figures/myPlot.png")
    plt.show()
    return "Yıllara Göre Toplam Satış grafiği"
def top_musteri():
    customer_sales = df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False)[:20]
    customer_sales.plot(kind='bar', figsize=(10, 6))
    plt.title('İlk 20 Müşteri')
    plt.xlabel('Müşteri Adı')
    plt.ylabel('Satış Miktarı')
    plt.savefig("figures/topMusteri.png")
    plt.show()
    return "İlk 20 Müşteri"

def getPredict(year, productName):
  return " " + year + " yılında " +productName+" satış tahmini 789909809'dır"


def top_grossing_cities():
    customer_sales = df.groupby('City')['Sales'].sum().sort_values(ascending=False)[:20]
    customer_sales.plot(kind='bar', figsize=(10, 6))
    plt.title('İlk 20 Şehir')
    plt.xlabel('Şehirler')
    plt.ylabel('Satış Miktarı')
    plt.savefig("figures/topcity.png")
    plt.show()
    return "İlk 20 Şehir"

def urun_kategori():
    customer_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    customer_sales.plot(kind='bar', figsize=(10, 6))
    plt.title('Kategori-Satış')
    plt.xlabel('Kategoriler')
    plt.ylabel('Satış Miktarı')
    plt.savefig("figures/urunkategori.png")
    plt.show()
    return "Kategori-Satış"

def calisan_dagilimi():
    sales_by_year = df2.groupby('team')['no_of_workers'].sum()
    plt.bar(sales_by_year.index, sales_by_year.values)
    plt.title('Departmanlara göre çalışan sayısı')
    plt.xlabel('Departman')
    plt.ylabel('Çalışan sayısı')
    plt.savefig("figures/calisandagilimi.png")
    plt.show()
    return "Departmanlara göre çalışan sayısı"
def actual_productivity_rate():
    sales_by_year = df2.groupby('team')['actual_productivity'].sum()
    plt.bar(sales_by_year.index, sales_by_year.values)
    plt.title('Actual productivity by team')
    plt.xlabel('Team')
    plt.ylabel('Actual productivity')
    plt.savefig("figures/productivity.png")
    plt.show()
    return "Verimlilik"

def Attrcalisan_yas_dagilim():
    # Attrition değeri "Yes" olan çalışanları filtrele ve yaşa göre sırala
    attrition_yes = calisan[calisan['Attrition'] == 'Yes']
    attrition_yes_sorted = attrition_yes.sort_values(by='Age')

    plt.bar(attrition_yes_sorted['Age'], range(attrition_yes_sorted.shape[0]))
    plt.xlabel('Yaş')
    plt.ylabel('Çalışan Sayısı')
    plt.title('Çalışanların Yaş Dağılımı')
    plt.show()
    return "Yıpranmış olan çalışanlar"

def calisan_yas_dagilim():
    # Attrition değeri "No" olan çalışanları filtrele ve yaşa göre sırala
    attrition_no = calisan[calisan['Attrition'] == 'No']
    attrition_no_sorted = attrition_no.sort_values(by='Age')

    plt.bar(attrition_no_sorted['Age'], range(attrition_no_sorted.shape[0]))
    plt.xlabel('Yaş')
    plt.ylabel('Çalışan Sayısı')
    plt.title('Çalışanların Yaş Dağılımı')
    plt.show()
    return "Çalışan yaş dağılımı"

def attrition_yas_aralik():
    # Yaş aralıklarını tanımla
    age_ranges = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65)]

    # Yaş aralığına göre gruplayarak Attrition değerlerini say ve oranı hesapla
    attrition_ratios = []
    for age_range in age_ranges:
        age_min, age_max = age_range
        attrition_age_range = calisan[(calisan['Age'] >= age_min) & (calisan['Age'] < age_max)]
        attrition_count = attrition_age_range['Attrition'].value_counts()

        attrition_ratio = 0
        if 'Yes' in attrition_count:
            attrition_ratio = attrition_count['Yes'] / attrition_count.sum()
        attrition_ratios.append(attrition_ratio)

    age_labels = ['{}-{}'.format(age_range[0], age_range[1]) for age_range in age_ranges]
    plt.bar(age_labels, attrition_ratios)
    plt.xlabel('Yaş Aralığı')
    plt.ylabel('Attrition Oranı')
    plt.title('Yaş Aralığına Göre Attrition Oranları')
    plt.xticks(rotation=45)
    plt.show()
    return "Yaş aralığına göre gruplayarak Attrition değerleri"

