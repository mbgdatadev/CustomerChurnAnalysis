import pandas as pd
import matplotlib.pyplot as plt

# RFM verilerini yükle
rfm_data = pd.read_csv("rfm.csv", index_col=0)

# Örnek olarak 'hibernating' ve 'about_to_sleep' segmentlerini churn olarak tanımlayalım
churn_segments = ['hibernating', 'about_to_sleep']

# Churn müşterileri
churn_customers = rfm_data[rfm_data['segment'].isin(churn_segments)]

# Toplam müşteri sayısı
total_customers = rfm_data.shape[0]

# Churn müşteri sayısı
churn_customer_count = churn_customers.shape[0]

# Churn oranı
churn_rate = churn_customer_count / total_customers
print(f"Churn Oranı: {churn_rate:.2%}")

# Her segmentin müşteri sayısını hesapla
segment_counts = rfm_data['segment'].value_counts()

# Her segmentin oranını hesapla
segment_percentages = segment_counts / total_customers * 100

# DataFrame oluştur
segment_df = pd.DataFrame({
    'Segment': segment_percentages.index,
    'Percentage': segment_percentages.values
})

# Pasta grafiği oluştur
plt.figure(figsize=(12, 8))
colors = plt.get_cmap('tab20').colors  # Her segment için farklı renkler
wedges, texts, autotexts = plt.pie(
    segment_df['Percentage'], 
    labels=segment_df['Segment'], 
    autopct='%1.1f%%', 
    colors=colors, 
    startangle=140,
    pctdistance=0.85
)

# Tek bir legend'da her iki bilgiyi ekleyin
handles = wedges + [plt.Line2D([0], [0], color='grey', lw=4)]
labels = [f"{s}: {p:.1f}%" for s, p in zip(segment_df['Segment'], segment_df['Percentage'])] + [f"**Churn Rate:** {churn_rate:.2%}"]
plt.legend(handles, labels, title="Segments Rate", loc="center left", bbox_to_anchor=(1.05, 0.5,0.5,0.5))



plt.title('Customer Segments Rate ')
plt.show()
