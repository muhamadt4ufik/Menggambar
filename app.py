"""Dashboard hitung kendaraan yang mendekati zebra cross."""
from collections import defaultdict, deque
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "CCTV JL Gerbang Pemuda.mp4"
TRACKER_PATH = ROOT / "bytetrack_vehicle.yaml"

st.set_page_config(page_title="Hitung Motor · Gerbang Pemuda", page_icon="🏍️", layout="wide")
st.markdown("""
<style>
.stApp { background: #f4f7fb; }
[data-testid="stMetric"] { background: white; border: 1px solid #e5eaf2; padding: 16px; border-radius: 14px; }
div[data-testid="stVideo"] { border-radius: 14px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🏍️ Dashboard Hitung Motor — JL Gerbang Pemuda")
st.caption("Deteksi dan hitung motor yang bergerak mendekati zebra cross. Kendaraan yang menjauh diabaikan.")

if not VIDEO_PATH.exists():
    st.error(f"Video tidak ditemukan: {VIDEO_PATH.name}. Letakkan video di folder yang sama dengan app.py.")
    st.stop()

st.subheader("Video sumber")
st.video(str(VIDEO_PATH), autoplay=True, muted=True)

with st.sidebar:
    st.header("Pengaturan analisis")
    st.write(f"**Video:** {VIDEO_PATH.name}")
    model_choice = st.selectbox(
        "Model deteksi",
        ["YOLOv8s — lebih akurat, lebih lambat", "YOLOv8n — lebih ringan"],
        help="YOLOv8s biasanya lebih baik untuk motor yang terlihat sebagian, tetapi pemrosesan lebih lambat.",
    )
    model_file = "yolov8s.pt" if model_choice.startswith("YOLOv8s") else "yolov8n.pt"
    model_path = ROOT / model_file
    conf = st.slider("Ambang deteksi (lebih rendah = lebih sensitif)", 0.05, 0.6, 0.12, 0.01)
    st.markdown("**Zona hitung dekat zebra cross**")
    st.caption("Zona awal ditandai di kiri bawah. Kendaraan dihitung saat memasuki zona setelah terdeteksi bergerak ke arahnya.")
    zone_x = st.slider("Batas kanan zona (%)", 10, 60, 40, 1) / 100
    zone_y = st.slider("Batas atas zona (%)", 30, 85, 48, 1) / 100
    st.caption("Sesuaikan batas zona jika lokasi zebra cross pada video berbeda.")
    run = st.button("▶  Mulai analisis", type="primary", use_container_width=True)

st.info("Deteksi hanya untuk **motor** dan ditandai kotak **hijau**. Label `MENUJU` menandakan arah mendekati zebra cross. Mobil diabaikan. Semua frame dianalisis agar hasil video diputar dengan lancar; proses analisis dapat memakan waktu.")

if run:
    try:
        model_source = str(model_path) if model_path.exists() else model_file
        model = YOLO(model_source)
    except Exception as exc:
        st.error(f"Model deteksi tidak dapat dimuat: {exc}")
        st.stop()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zone = [(0, int(height * zone_y)), (int(width * zone_x), int(height * zone_y)),
            (int(width * zone_x), height), (0, height)]
    history = defaultdict(lambda: deque(maxlen=15))
    approaching = {}
    counted = set()
    events = []
    count_motor = 0
    progress = st.progress(0, text="Memulai analisis…")
    st.subheader("Pratinjau deteksi motor")
    st.caption("Kotak hijau akan muncul pada frame saat motor terdeteksi.")
    live_preview = st.empty()
    video_slot = st.empty()
    status = st.empty()
    out_path = Path(tempfile.gettempdir()) / "dashboard_kendaraan_gerbang_pemuda.mp4"
    # Tulis satu frame keluaran untuk setiap frame sumber agar video hasil tetap pada FPS aslinya.
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_i = 0
    started = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, tracker=str(TRACKER_PATH), classes=[3],
                             conf=conf, verbose=False)[0]
        annotated = frame.copy()
        cv2.rectangle(annotated, (0, int(height*zone_y)), (int(width*zone_x), height), (30, 180, 70), 2)
        cv2.putText(annotated, "ZONA ZEBRA CROSS", (12, int(height*zone_y)+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 190, 80), 2)
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            scores = boxes.conf.cpu().numpy()
            for box, track_id, cls_id, score in zip(xyxy, ids, classes, scores):
                x1, y1, x2, y2 = box.tolist()
                foot = ((x1+x2)//2, y2)
                hist = history[track_id]
                hist.append(foot)
                if len(hist) >= 3:
                    # Kendaraan mendekati zebra cross dari arah jalan (kanan/atas ke kiri/bawah).
                    dx, dy = hist[-1][0]-hist[0][0], hist[-1][1]-hist[0][1]
                    # Pertahankan status arah setelah terkonfirmasi agar tertutup sesaat tidak
                    # membuat kendaraan kehilangan status mendekat.
                    approaching[track_id] = approaching.get(track_id, False) or (dy > 2 and dx < 45)
                label = "Motor"
                color = (0, 210, 60)  # BGR hijau
                inside = cv2.pointPolygonTest(np.array(zone, dtype="int32"), foot, False) >= 0
                if inside and approaching.get(track_id, False) and track_id not in counted:
                    counted.add(track_id)
                    count_motor += 1
                    events.append({"Waktu (video)": f"{frame_i/fps:.1f} dtk", "ID": int(track_id),
                                   "Keyakinan": round(float(score), 2)})
                direction = "MENUJU" if approaching.get(track_id, False) else "—"
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 3)
                cv2.putText(annotated, f"{label} #{track_id} {direction}", (x1,max(25,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        writer.write(annotated)
        # Tampilkan frame beranotasi selama analisis agar deteksi terlihat langsung.
        if frame_i % max(1, int(fps / 6)) == 0:
            preview_width = 960
            preview_height = int(height * preview_width / width)
            preview_frame = cv2.resize(annotated, (preview_width, preview_height))
            live_preview.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB),
                               caption=f"Frame {frame_i/fps:.1f} dtk", use_container_width=True)
        # Perbarui elemen Streamlit seperlunya; merender setiap frame membuat UI tersendat.
        if frame_i % max(1, int(fps * 3)) == 0:
            progress.progress(min(frame_i / max(1,total), 1.0),
                              text=f"Memproses {frame_i/fps:.0f} dtk dari {total/fps:.0f} dtk…")
            status.caption(f"ID dihitung: {len(counted)} · Waktu berjalan: {time.time()-started:.0f} dtk")
        frame_i += 1
    cap.release()
    writer.release()
    progress.progress(1.0, text="Analisis selesai")
    st.success(f"Selesai dalam {time.time()-started:.0f} detik. Total motor menuju zebra cross: {count_motor}.")

    st.metric("🏍️ Total motor menuju zebra cross", count_motor)
    st.subheader("Video dengan deteksi")
    if out_path.exists():
        video_slot.video(str(out_path), autoplay=True, muted=True)
        st.download_button("Unduh video hasil analisis", out_path.read_bytes(),
                           file_name="kendaraan_menuju_zebra_cross.mp4", mime="video/mp4")
    st.subheader("Riwayat kendaraan yang dihitung")
    if events:
        df = pd.DataFrame(events)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Unduh riwayat CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="hitungan_kendaraan.csv", mime="text/csv")
    else:
        st.warning("Tidak ada motor yang masuk zona dengan arah mendekat. Coba turunkan ambang deteksi atau sesuaikan zona.")
else:
    st.markdown("### Ringkasan")
    a, b = st.columns(2)
    a.metric("Kategori deteksi", "Motor saja")
    b.metric("Aturan hitung", "Arah mendekat saja")
    st.caption("Tekan **Mulai analisis** untuk memproses video, melihat kotak deteksi, dan mendapatkan rekap serta CSV.")
