# Dashboard hitung kendaraan

## Menjalankan

Buka PowerShell di folder ini, lalu jalankan:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Dashboard akan terbuka di browser dan video sumber mulai diputar tanpa suara. Tekan **Mulai analisis** untuk melihat pratinjau deteksi langsung dengan kotak hijau. Pilih YOLOv8s untuk deteksi yang lebih kuat (lebih lambat) atau YOLOv8n untuk pemrosesan lebih ringan; bobot yang belum tersedia akan diunduh satu kali. Dashboard hanya mendeteksi motor dan menghitung tiap ID sekali ketika motor bergerak mendekati dan masuk zona. Mobil dan motor yang terdeteksi menjauh tidak dihitung. Ambang deteksi awal dibuat lebih sensitif dan pelacak mempertahankan ID lebih lama saat tertutup sesaat. Setelah selesai, video beranotasi akan diputar otomatis tanpa suara.

## Kalibrasi zona

Zona awal berada di kiri bawah frame, dekat zebra cross yang terlihat pada video. Gunakan pengaturan **Batas kanan zona** dan **Batas atas zona** di sidebar bila perlu memperluas area hitung. Motor dihitung saat titik roda/bawah kotak masuk zona setelah riwayat geraknya menunjukkan arah ke kiri dan ke bawah. Analisis semua frame membutuhkan waktu lebih lama, tetapi video hasil disimpan pada frame rate asli agar pemutaran lebih lancar.

Hasil mencakup video beranotasi dan riwayat motor dalam CSV. Motor yang tertutup sepenuhnya oleh kendaraan lain tidak dapat dideteksi sampai muncul kembali; ambang yang lebih sensitif juga dapat meningkatkan deteksi keliru.
