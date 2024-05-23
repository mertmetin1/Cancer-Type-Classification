from matplotlib import pyplot as plt

import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import xgboost as xgb

# Verileri yükleme
data = pd.read_csv("data.csv")
labels = pd.read_csv("labels.csv")

# Veri setinin genel bilgilerini kontrol et
print("Veri Seti Başlıkları ve İlk 3 Satır:")
print(data.head(3))
print()
print("\nVeri Seti Bilgileri:")
print(data.info())
print()
print("\nVeri Seti İstatistikleri:")
print(data.describe())
print()
print("\nEksik Değerlerin Sayısı:")
print(data.isnull().sum())
print()
print("\n'Sutun2' Benzersiz Değerleri:")
print(data.nunique())
print()
# Veri setinin genel bilgilerini kontrol et
print("Veri Seti Başlıkları ve İlk 3 Satır:")
print(labels.head(3))
print()
print("\nVeri Seti Bilgileri:")
print(labels.info())
print()
print("\nVeri Seti İstatistikleri:")
print(labels.describe())
print()
print("\nEksik Değerlerin Sayısı:")
print(labels.isnull().sum())
print()
print("\n'Sutun2' Benzersiz Değerleri:")
print(labels.nunique())
print()


# Özellik ve hedef değişkenlerini ayır
X = data.drop(columns=['Sample']) 
y = labels.drop(columns=['Sample']) 

# LabelEncoder oluştur
label_encoder_y = LabelEncoder()

# Label encoding uygula
y_encoded = label_encoder_y.fit_transform(y)

# Reshape y to 1D array
y = y.values.ravel()

# Min-max normalizasyon uygula
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


#ilk çalıştırdğında yorum satırındaki kodları saçılırtır sonraki seferlerde tekrar yorum satırına alabilirsin 
"""

print("RFE ile özellik seçimi (10 özellik)")

# RFE ile en iyi 10 özelliği seç
num_features_to_select = 10
rfe = RFE(estimator=LinearDiscriminantAnalysis(), n_features_to_select=num_features_to_select)
rfe.fit(X_scaled, y)

# Seçilen özelliklerin indekslerini al
selected_features_idx = rfe.support_

# Seçilen özelliklerin isimlerini al
selected_features = X.columns[selected_features_idx]

# Seçili özellikleri içeren yeni veri setini oluştur
selected_data = X[selected_features]

# Seçili özellikleri içeren yeni veri setini CSV dosyasına kaydet
selected_data.to_csv("selected_features.csv", index=False)
"""
selected_features = pd.read_csv("selected_features.csv")
X=selected_features


# Korelasyon matrisi ile seçilen özellikleri görüntüle
df_rfe_cm = selected_features.corr()          #selected_data yerine  selected_features  yaz 
print("\nKorelasyon Matrisi:")
print(df_rfe_cm)

# Korelasyon matrisini görselleştir
plt.figure(figsize=(10, 8))
sns.heatmap(df_rfe_cm, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Özellikler Arasındaki Korelasyon Matrisi')
plt.xlabel('Özellikler')
plt.ylabel('Özellikler')
plt.show()





# Veriyi eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

def calculate_metrics(cm):
    metrics = []
    for i in range(len(cm)):
        tn = sum(cm[j][k] for j in range(len(cm)) for k in range(len(cm)) if j != i and k != i)
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(len(cm)) if j != i)
        fn = sum(cm[i][j] for j in range(len(cm)) if j != i)
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        metrics.append((tp, fp, tn, fn, sensitivity, specificity))
    
    return metrics

##################################################################################################
#RandomForestClassifier
print("###################################################### RandomForestClassifier #############################################")


# RandomForestClassifier modelini eğit
rf_model = RandomForestClassifier(n_estimators=4, random_state=42)
rf_model.fit(X_train, y_train)

# Test kümesi üzerinde tahmin yap
rf_y_pred = rf_model.predict(X_test)

# Calculate confusion matrix
rf_cm = confusion_matrix(y_test, rf_y_pred)

print()
# Confusion matrixi yazdır
print(" RF Confusion Matrix:")
print(rf_cm)

plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title('RF Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Random Forest için metrics hesapla
rf_metrics = calculate_metrics(rf_cm)

# Random Forest için metrikleri yazdır
print()
print("Random Forest Metrics:")
for i, metric in enumerate(rf_metrics):
    print(f"Random Forest Class {i+1}:")
    print(f"TP: {metric[0]}, FP: {metric[1]}, TN: {metric[2]}, FN: {metric[3]}, Sensitivity: {metric[4]:.2f}, Specificity: {metric[5]:.2f}")
    print()


rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(" RF Accuracy:", rf_accuracy)
print()



##################################################################################################
#XGBoost
print("###################################################### XGBoost #############################################")


# XGBoost modelini oluştur
xgb_model = xgb.XGBClassifier()

# Modeli eğit
xgb_model.fit(X_train, y_train)

# Test verileri üzerinde tahmin yap
XGB_y_pred = xgb_model.predict(X_test)

# Calculate confusion matrix
XGB_cm = confusion_matrix(y_test, XGB_y_pred)

print()
# Confusion matrixi yazdır
print("XGB Confusion Matrix:")
print(XGB_cm)

plt.figure(figsize=(8, 6))
sns.heatmap(XGB_cm, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Random Forest için metrics hesapla
XGB_metrics = calculate_metrics(XGB_cm)


print()
# XGBoost için metrikleri yazdır
print("XGBoost Metrics:")
for i, metric in enumerate(XGB_metrics):
    print(f"XGBoost Class {i+1}:")
    print(f"TP: {metric[0]}, FP: {metric[1]}, TN: {metric[2]}, FN: {metric[3]}, Sensitivity: {metric[4]:.2f}, Specificity: {metric[5]:.2f}")
    print()
    
# Modelin doğruluğunu değerlendir
XGB_accuracy = accuracy_score(y_test, XGB_y_pred)
print("XGB Accuracy:", XGB_accuracy)
print()




    
#####################################################################################################



#####################################################################################################
