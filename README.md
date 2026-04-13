# Sürücü Dikkat İzleme (MVP)

Python ile geliştirilmiş, **gerçek zamanlı Sürücü Dikkat İzleme Sistemi**:

- **OpenCV**: webcam görüntüsü + ekran üstü yazı/çizim (UI)
- **MediaPipe Face Landmarker**: yüz landmark (işaret noktası) tespiti
- **EAR (Eye Aspect Ratio)**: göz kapanması / uyuklama algılama
- **Basit baş yönü sezgisi**: SOL / SAĞ / AŞAĞI

Bu proje **basit ve çalışan bir MVP**’dir (model eğitimi yok).

## 1) Sistem nasıl çalışır?

### Webcam + Yüz Landmark’ları
- OpenCV webcaminizden kareleri (frame) okur.
- MediaPipe yüz landmark noktalarını her karede tespit eder.
- Landmark’lar normalize koordinatlardan \([0..1]\) piksel koordinatlarına çevrilir.

### Eye Aspect Ratio (EAR)
- Her göz için 6 landmark noktası alınır ve şu oran hesaplanır:

\[
EAR = \frac{\|p2-p6\| + \|p3-p5\|}{2\cdot\|p1-p4\|}
\]

- Göz kapandıkça dikey mesafeler küçülür → **EAR düşer**.
- EAR, **eşik değerin altında** belirli bir süre (örn. 1.2 sn) kalırsa sistem **DROWSY** durumunu verir.

### Baş Pozisyonu (basit)
Baş yönünü kaba bir şekilde, **burun konumu sezgisi** ile tahmin ederiz:
- Burun x konumunu yüzün sol/sağ referanslarının orta noktasıyla kıyaslarız → **SOL/SAĞ**
- Burun y konumunu alın bölgesindeki bir referansla kıyaslarız → **AŞAĞI**

### Dikkat Skoru + Durum
0–100 arası basit bir skor hesaplanır:
- göz kapanması (EAR eşik altı → ceza)
- baş yönü (SOL/SAĞ/AŞAĞI → ceza)

Sonuç olarak ekranda şunlardan biri gösterilir:
- **ATTENTIVE** (dikkatli)
- **DISTRACTED** (dalgın/baş farklı yönde veya yüz bulunamadı)
- **DROWSY** (uyuklama algılandı)

Uyuklama varsa ekrana uyarı yazısı basılır.

## 2) Çalıştırma (adım adım)

### Adım A — Sanal ortam oluşturma ve aktif etme (önerilir)
Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Adım B — Bağımlılıkları yükleme

```bash
pip install -r requirements.txt
```

### Adım C — Uygulamayı çalıştırma

```bash
python main.py
```

Kontroller:
- Çıkmak için **Q**
- Çıkmak için **Esc**

Kamera açılmazsa `main.py` içindeki `camera_index` değerini değiştirip deneyin (0, 1, 2...).

## 3) Daha sonra neler geliştirilebilir?

- **Kullanıcıya göre kalibrasyon**: farklı yüz şekilleri/kameralar için EAR ve baş eşiğini otomatik ayarlama.
- **Daha iyi baş pozu**: sezgisel yaklaşımı `solvePnP` ile daha gerçekçi yaw/pitch/roll hesabına çevirmek.
- **Göz kırpma vs uyuklama ayrımı**: kısa blink ile uzun süre kapanmayı daha sağlam ayırmak (örn. blink hızı).
- **Dayanıklılık**: gözlük, düşük ışık, örtülme durumları; yüz ROI takibi ve zaman içinde smoothing.
- **Uyarılar**: sesli uyarı, olay kaydı (log), veya başka bir sisteme bildirim gönderme.
- **Çoklu yüz**: sürücü/yolcu ayrımı (şu an ilk bulunan yüz kullanılıyor).

