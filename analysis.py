import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# Veri setini yükle
data_ = df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2009-2010")

data=data_.copy()


def create_rfm(dataframe, csv=False):
 
    # VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # RFM skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm

rfm_data=create_rfm(dataframe=data,csv=True)


# Ülke bazında toplam harcamaları görselleştir
plt.figure(figsize=(12, 6))
country_data = data.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
sns.barplot(x=country_data.index, y=country_data.values)
plt.title('Total Spending by Country')
plt.xticks(rotation=90)
plt.ylabel('Total Spending')
plt.xlabel('Country')
plt.show()


# Grafik 2: Recency, Frequency ve Monetary Dağılımları
plt.figure(figsize=(18, 12))

# Recency Dağılımı
plt.subplot(3, 1, 1)
sns.histplot(rfm_data['recency'], bins=50, kde=True)
plt.title('Recency Distribution')

# Frequency Dağılımı
plt.subplot(3, 1, 2)
sns.histplot(rfm_data['frequency'], bins=50, kde=True)
plt.title('Frequency Distribution')

# Monetary Dağılımı
plt.subplot(3, 1, 3)
sns.histplot(rfm_data['monetary'], bins=50, kde=True)
plt.title('Monetary Distribution')

plt.tight_layout()
plt.show()

# RFM Segmentleri ile İlgili İstatistikler Tablosu
segment_stats = rfm_data.groupby('segment').agg({
    'recency': ['mean', 'std', 'min', 'max'],
    'frequency': ['mean', 'std', 'min', 'max'],
    'monetary': ['mean', 'std', 'min', 'max'],
    'segment': 'count'
}).reset_index()

segment_stats.columns = ['Segment', 'Recency Mean', 'Recency Std', 'Recency Min', 'Recency Max',
                         'Frequency Mean', 'Frequency Std', 'Frequency Min', 'Frequency Max',
                         'Monetary Mean', 'Monetary Std', 'Monetary Min', 'Monetary Max',
                         'Customer Count']

print(segment_stats)

# Segmentlerin görselleştirilmesi
plt.figure(figsize=(12, 8))
sns.heatmap(segment_stats.set_index('Segment').drop('Customer Count', axis=1).astype(float), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('RFM Segment Statistics Heatmap')
plt.show()

# Grafik : RFM Segmentlerinin Dağılımı
plt.figure(figsize=(10, 6))
sns.countplot(x='segment', data=rfm_data, order=rfm_data['segment'].value_counts().index, palette='viridis')
plt.xticks(rotation=45)
plt.title('RFM Segment Counts')
plt.xlabel('Segments')
plt.ylabel('Count')
plt.savefig('rfm_segment_counts.png')
plt.show()