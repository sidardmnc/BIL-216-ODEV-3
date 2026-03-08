import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import medfilt

# ==========================================
# 1. SES DOSYASINI OKUMA VE NORMALİZASYON
# ==========================================
dosya_adi = 'ses_kaydi.wav'  # Kendi ses kaydının adını buraya yaz
fs, signal = wavfile.read(dosya_adi)

# Eğer ses stereo ise tek kanala (mono) düşürüyoruz
if len(signal.shape) > 1:
    signal = signal[:, 0]

# Sinyali float tipine çevirip [-1, 1] aralığına normalize ediyoruz (Ödev İsteri 4.A)
signal = signal.astype(np.float32)
signal = signal / np.max(np.abs(signal))

# ==========================================
# 2. PENCERELEME (FRAMING) VE OVERLAP
# ==========================================
frame_duration = 0.02  # 20 ms pencere
overlap_duration = 0.01  # 10 ms örtüşme (%50 overlap)

frame_len = int(frame_duration * fs)
step = int(overlap_duration * fs)

# Sızıntıyı (leakage) önlemek için Hamming penceresi oluşturuyoruz
hamming_window = np.hamming(frame_len)

num_frames = (len(signal) - frame_len) // step + 1

energies = np.zeros(num_frames)
zcrs = np.zeros(num_frames)

# Her bir pencere için Enerji ve ZCR hesaplaması
for i in range(num_frames):
    start = i * step
    end = start + frame_len

    # Pencereyi sinyalden alıp Hamming penceresi ile çarpıyoruz
    frame = signal[start:end] * hamming_window

    # Kısa Süreli Enerji (Karesel)
    energies[i] = np.sum(frame ** 2)

    # Sıfır Geçiş Oranı (ZCR)
    # İşaret değişimlerini sayıyoruz
    zcrs[i] = 0.5 * np.sum(np.abs(np.sign(frame[1:]) - np.sign(frame[:-1])))

# ==========================================
# 3. DİNAMİK GÜRÜLTÜ EŞİĞİ VE VAD KARARI
# ==========================================
# İlk 150 ms'lik (0.15 sn) kısmın sessizlik/gürültü olduğunu varsayıyoruz
noise_duration = 0.15
noise_frames = int(noise_duration / overlap_duration)

# Gürültü tabanını (noise floor) hesaplama
noise_energy_mean = np.mean(energies[:noise_frames])

# Dinamik Eşik: Ortalama gürültü enerjisinin 3 katı (İhtiyaca göre 2-5 arası ayarlanabilir)
energy_threshold = noise_energy_mean * 3

# Eşiği geçenleri 1 (Konuşma), geçemeyenleri 0 (Sessizlik) yapıyoruz
vad = (energies > energy_threshold).astype(int)

# ==========================================
# 4. HANGOVER (AKICILIK) KONTROLÜ
# ==========================================
# Konuşma sırasındaki mikro duraksamaları (Sessiz harf patlamaları vb.) 
# silmemek için 5 birimlik Medyan Filtre uyguluyoruz. 
# Bu, ödevdeki "Hangover Time" işlevini kusursuz yerine getirir.
vad_smoothed = medfilt(vad, kernel_size=5)

# ==========================================
# 5. VOICED (SESLİ) / UNVOICED (SESSİZ) AYRIMI
# ==========================================
# Sadece konuşma olan pencerelerin ZCR ortalamasını bulalım ki eşiğimiz dinamik olsun
mean_zcr_speech = np.mean(zcrs[vad_smoothed == 1])

voiced = np.zeros(num_frames)
unvoiced = np.zeros(num_frames)

for i in range(num_frames):
    if vad_smoothed[i] == 1:
        # Mantıksal Sınıflandırıcı: Enerji yüksek ve ZCR görece düşükse -> VOICED
        if zcrs[i] < mean_zcr_speech * 1.5 and energies[i] > energy_threshold * 1.2:
            voiced[i] = 1
        else:
            # ZCR yüksek ve enerji orta/düşük seviyedeyse -> UNVOICED
            unvoiced[i] = 1

# ==========================================
# 6. SESSİZLİĞİ ATIP YENİ DOSYA OLUŞTURMA
# ==========================================
# Sinyalin sadece konuşma olan kısımlarını maskelemek için sample bazında bir maske oluşturuyoruz
sample_vad = np.zeros(len(signal))
for i in range(num_frames):
    if vad_smoothed[i] == 1:
        sample_vad[i * step: i * step + frame_len] = 1

# Sadece konuşma (1) olan indeksleri alıp uç uca ekliyoruz
clean_signal = signal[sample_vad == 1]

# Sıkıştırma oranını hesaplayıp konsola yazdırıyoruz
orijinal_sure = len(signal) / fs
yeni_sure = len(clean_signal) / fs
sikistirma_orani = (1 - (yeni_sure / orijinal_sure)) * 100

print(f"Orijinal Süre: {orijinal_sure:.2f} saniye")
print(f"Temizlenmiş Süre: {yeni_sure:.2f} saniye")
print(f"Sıkıştırma Başarısı: %{sikistirma_orani:.2f} oranında sessizlik silindi.")

# Temizlenmiş (Sessizliği atılmış) dosyayı kaydet
wavfile.write('ODEV3_temizlenmis_ses.wav', fs, np.int16(clean_signal * 32767))

# ==========================================
# 7. GÖRSELLEŞTİRME VE PLOTTING
# ==========================================
t_signal = np.arange(len(signal)) / fs
t_frames = np.arange(num_frames) * step / fs

# Grafik maskeleri için sample bazlı voiced/unvoiced dizileri
sample_voiced = np.zeros(len(signal))
sample_unvoiced = np.zeros(len(signal))

for i in range(num_frames):
    if voiced[i] == 1:
        sample_voiced[i * step: i * step + frame_len] = 1
    elif unvoiced[i] == 1:
        sample_unvoiced[i * step: i * step + frame_len] = 1

plt.figure(figsize=(14, 10))

# 1. Grafik: Orijinal Sinyal
plt.subplot(3, 1, 1)
plt.plot(t_signal, signal, color='gray')
plt.title('1. Orijinal Ses Sinyali (Zaman Domeni)')
plt.ylabel('Genlik')
plt.grid(True, linestyle='--', alpha=0.6)

# 2. Grafik: Enerji ve ZCR (Aynı eksende görebilmek için kendi içlerinde normalize edildiler)
plt.subplot(3, 1, 2)
plt.plot(t_frames, energies / np.max(energies), label='Kısa Süreli Enerji (Normalize)', color='blue')
plt.plot(t_frames, zcrs / np.max(zcrs), label='ZCR (Normalize)', color='red', alpha=0.7)
plt.title('2. Pencere Bazlı Enerji ve Sıfır Geçiş Oranı (ZCR)')
plt.ylabel('Normalize Değerler')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 3. Grafik: Voiced / Unvoiced Bölgeleri Renklendirilmiş Sinyal
plt.subplot(3, 1, 3)
plt.plot(t_signal, signal, color='gray', alpha=0.4)
plt.fill_between(t_signal, -1, 1, where=(sample_voiced == 1), color='green', alpha=0.4, label='Voiced (Sesli: A, O, U)')
plt.fill_between(t_signal, -1, 1, where=(sample_unvoiced == 1), color='gold', alpha=0.5,
                 label='Unvoiced (Sessiz: S, Ş, F)')
plt.title('3. VAD ve Voiced/Unvoiced (V/UV) Ayrımı')
plt.xlabel('Zaman (Saniye)')
plt.ylabel('Genlik')
plt.legend()

plt.tight_layout()
plt.show()