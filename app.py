import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time
from collections import Counter

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kopi Vision — Deteksi Kematangan Buah Kopi",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #0d0d0d;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(139,90,43,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(185,134,75,0.12) 0%, transparent 60%);
}

header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

.hero-banner {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(185,134,75,0.2);
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 900;
    color: #f5e6c8;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.hero-title span { color: #c8882a; }
.hero-sub {
    color: #a08060;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(185,134,75,0.15);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}
.card-warn {
    background: rgba(200,100,30,0.07);
    border: 1px solid rgba(200,100,30,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}

.stat-row { display: flex; gap: 0.8rem; margin-top: 1.2rem; flex-wrap: wrap; }
.stat-box {
    flex: 1; min-width: 100px;
    background: rgba(200,136,42,0.08);
    border: 1px solid rgba(200,136,42,0.25);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 700;
    color: #c8882a; line-height: 1;
}
.stat-label {
    font-size: 0.72rem; color: #8a6a40;
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem;
}

.det-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(200,136,42,0.12);
    border: 1px solid rgba(200,136,42,0.3);
    border-radius: 999px; padding: 0.35rem 0.9rem;
    margin: 0.25rem 0.2rem; font-size: 0.85rem;
    color: #e0c080; font-weight: 500;
}
.conf-dot { width: 8px; height: 8px; border-radius: 50%; background: #c8882a; flex-shrink: 0; }

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; color: #f0dbb0;
    margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(200,136,42,0.4), transparent);
    margin: 1.5rem 0;
}

/* Override badge — warna berbeda untuk kelas yang di-override */
.override-badge {
    display: inline-block;
    background: rgba(80,160,120,0.15);
    border: 1px solid rgba(80,200,120,0.3);
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.72rem;
    color: #80e0a0;
    margin-left: 0.4rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #8b4a1a, #c8882a);
}
section[data-testid="stSidebar"] {
    background: #0a0a0a !important;
    border-right: 1px solid rgba(185,134,75,0.15);
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
.stButton > button {
    background: linear-gradient(135deg, #8b4a1a, #c8882a) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.04em !important;
    padding: 0.55rem 1.8rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 18px rgba(200,136,42,0.25) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important; transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(200,136,42,0.4) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(200,136,42,0.35) !important;
    border-radius: 12px !important;
}
[data-testid="caption"] { color: #7a5c30 !important; font-size: 0.78rem !important; text-align: center !important; }
.stAlert { border-radius: 10px !important; }
.stSpinner > div { border-top-color: #c8882a !important; }
[data-testid="stRadio"] label { color: #c8a060 !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO('runs/detect/runs/detect/kopi_yolo_test3/weights/best.pt')


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Kopi <span>Vision</span></div>
    <div class="hero-sub">☕ &nbsp; Sistem Deteksi Kematangan Buah Kopi &nbsp; ☕</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Playfair Display",serif; font-size:1.4rem;
                color:#f0dbb0; margin-bottom:1.2rem; padding-bottom:0.8rem;
                border-bottom:1px solid rgba(185,134,75,0.2);'>
        ⚙️ Panel Kontrol
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Mode Deteksi",
        ["📁 Upload Gambar", "🎥 Realtime Kamera"],
        label_visibility="collapsed",
    )

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Pengaturan Model Umum ──────────────────
    st.markdown("""<div style='color:#8a6a40; font-size:0.8rem; text-transform:uppercase;
                letter-spacing:0.07em; margin-bottom:0.5rem;'>Pengaturan Model</div>""",
                unsafe_allow_html=True)

    conf_threshold = st.slider("Confidence Minimum", 0.1, 1.0, 0.25, 0.05,
                                help="Abaikan deteksi di bawah nilai ini")
    iou_threshold  = st.slider("IoU Threshold (NMS)", 0.1, 1.0, 0.45, 0.05,
                                help="Threshold untuk Non-Max Suppression")

    if mode == "🎥 Realtime Kamera":
        img_size = st.select_slider("Ukuran Inferensi", [160, 320, 480, 640], value=320)
    else:
        img_size = st.select_slider("Ukuran Inferensi", [320, 480, 640, 800], value=640)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Load model & ambil nama kelas ─────────
    with st.spinner("Memuat model…"):
        try:
            model = load_model()
            model_ok = True
        except Exception as e:
            model_ok = False
            model_err = str(e)

    if not model_ok:
        st.error(f"Gagal memuat model:\n{model_err}")
        st.stop()

    class_names = list(model.names.values())

    # ── CLASS CONFIDENCE OVERRIDE ──────────────
    st.markdown("""
    <div style='color:#8a6a40; font-size:0.8rem; text-transform:uppercase;
                letter-spacing:0.07em; margin-bottom:0.4rem;'>
        🎛️ Koreksi Confidence per Kelas
    </div>
    <div style='color:#5a4020; font-size:0.75rem; line-height:1.5; margin-bottom:0.8rem;'>
        Naikkan nilai kelas yang sering salah terdeteksi
        (misal: <em>matang</em> terbaca sebagai <em>mentah</em>).
        Nilai di atas 1.0 = perbesar confidence, di bawah 1.0 = perkecil.
    </div>
    """, unsafe_allow_html=True)

    class_conf_multipliers = {}
    for cls_name in class_names:
        multiplier = st.slider(
            f"× {cls_name}",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.05,
            key=f"mult_{cls_name}",
            help=(
                f"Confidence semua deteksi '{cls_name}' akan dikali nilai ini "
                f"sebelum difilter. Default 1.0 = tidak ada perubahan."
            ),
        )
        class_conf_multipliers[cls_name] = multiplier

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Status Model ──────────────────────────
    st.markdown(f"""
    <div class='card' style='padding:1rem;'>
        <div style='color:#8a6a40; font-size:0.72rem; text-transform:uppercase;
                    letter-spacing:0.06em; margin-bottom:0.6rem;'>Status Model</div>
        <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;'>
            <div style='width:8px;height:8px;border-radius:50%;background:#4caf84;'></div>
            <span style='color:#c8e8b0; font-size:0.85rem; font-weight:500;'>Model Aktif</span>
        </div>
        <div style='color:#9a7040; font-size:0.8rem;'>
            🏷️ {len(class_names)} kelas:
            <span style='color:#c8a060;'>{", ".join(class_names)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUNGSI INTI: Terapkan Class Confidence Override
# ─────────────────────────────────────────────
def apply_class_conf_override(results, model, multipliers, global_conf_thresh):
    """
    Filter ulang deteksi setelah mengalikan confidence tiap kelas
    dengan multiplier-nya masing-masing.

    Kenapa ini membantu?
    - Model yang kurang data pada kelas tertentu (misal 'matang') cenderung
      menghasilkan confidence rendah untuk kelas itu.
    - Dengan menaikkan multiplier kelas tersebut, deteksi yang sebelumnya
      di-drop karena di bawah threshold bisa 'diselamatkan'.
    - Sebaliknya, kelas yang terlalu sering muncul salah bisa dikecilkan.
    """
    filtered = []
    for r in results:
        for box in r.boxes:
            cls_id   = int(box.cls[0])
            raw_conf = float(box.conf[0])
            label    = model.names[cls_id]

            mult          = multipliers.get(label, 1.0)
            adjusted_conf = min(raw_conf * mult, 1.0)   # cap di 1.0

            if adjusted_conf >= global_conf_thresh:
                filtered.append({
                    "label":        label,
                    "raw_conf":     raw_conf,
                    "adj_conf":     adjusted_conf,
                    "overridden":   mult != 1.0,
                    "box":          box,
                })
    return filtered


# ─────────────────────────────────────────────
# HELPER: Render hasil deteksi
# ─────────────────────────────────────────────
def render_results(results, model, multipliers, conf_thresh):
    detections = apply_class_conf_override(results, model, multipliers, conf_thresh)
    total = len(detections)

    if total == 0:
        st.markdown("""
        <div class='card' style='text-align:center; padding:2rem;'>
            <div style='font-size:2.5rem; margin-bottom:0.5rem;'>🔍</div>
            <div style='color:#8a6a40; font-size:0.95rem;'>
                Tidak ada objek terdeteksi.<br>
                Coba turunkan <em>Confidence Minimum</em> atau naikkan
                <em>Koreksi Confidence</em> pada kelas yang diharapkan.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    counts = Counter(d["label"] for d in detections)

    # ── Ringkasan ──────────────────────────────
    stat_html = "<div class='stat-row'>"
    stat_html += f"""
    <div class='stat-box'>
        <div class='stat-num'>{total}</div>
        <div class='stat-label'>Total Terdeteksi</div>
    </div>
    """
    for cls_name, cnt in counts.most_common():
        stat_html += f"""
        <div class='stat-box'>
            <div class='stat-num'>{cnt}</div>
            <div class='stat-label'>{cls_name}</div>
        </div>
        """
    stat_html += "</div>"

    st.markdown(f"""
    <div class='card'>
        <div class='section-title'>📊 Ringkasan Deteksi</div>
        {stat_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Peringatan jika ada kelas yang di-override ─
    overridden_classes = list({d["label"] for d in detections if d["overridden"]})
    if overridden_classes:
        st.markdown(f"""
        <div class='card-warn'>
            <span style='color:#e09040; font-weight:600;'>⚠️ Koreksi Aktif</span>
            <span style='color:#9a6030; font-size:0.85rem;'>
             — Confidence kelas <strong style='color:#e0a060;'>{", ".join(overridden_classes)}</strong>
             telah disesuaikan dengan multiplier yang kamu atur.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Badge per deteksi ──────────────────────
    badges_html = "<div style='margin-top:0.4rem;'>"
    for d in sorted(detections, key=lambda x: -x["adj_conf"]):
        color = "#4caf84" if d["adj_conf"] >= 0.7 else ("#c8882a" if d["adj_conf"] >= 0.4 else "#c04040")
        override_tag = (
            f"<span class='override-badge'>×{multipliers.get(d['label'],1.0):.2f}</span>"
            if d["overridden"] else ""
        )
        badges_html += f"""
        <span class='det-badge'>
            <span class='conf-dot' style='background:{color};'></span>
            {d['label']}{override_tag}
            &nbsp;<span style='color:#6a5030; font-size:0.78rem;'>{d['adj_conf']:.0%}</span>
        </span>
        """
    badges_html += "</div>"

    st.markdown(f"""
    <div class='card'>
        <div class='section-title'>🏷️ Detail Setiap Deteksi</div>
        {badges_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence chart ───────────────────────
    st.markdown("<div class='section-title' style='margin-top:1rem;'>📈 Confidence per Deteksi</div>",
                unsafe_allow_html=True)
    for d in sorted(detections, key=lambda x: -x["adj_conf"]):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.progress(d["adj_conf"], text=d["label"])
        with col2:
            st.markdown(
                f"<div style='color:#c8882a; font-weight:600; text-align:right; "
                f"padding-top:0.25rem;'>{d['adj_conf']:.1%}</div>",
                unsafe_allow_html=True,
            )
        with col3:
            if d["overridden"]:
                st.markdown(
                    f"<div style='color:#5a9070; font-size:0.75rem; text-align:right; "
                    f"padding-top:0.3rem;'>asli: {d['raw_conf']:.1%}</div>",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────
# MODE: UPLOAD GAMBAR
# ─────────────────────────────────────────────
if mode == "📁 Upload Gambar":

    st.markdown("<div class='section-title'>📁 Upload Gambar Buah Kopi</div>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Seret & lepas gambar, atau klik untuk memilih",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image    = Image.open(uploaded_file).convert("RGB")
        img_rgb  = np.array(image)               # RGB — untuk ditampilkan
        img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # BGR — untuk YOLO

        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("<div class='section-title'>🖼️ Gambar Input</div>",
                        unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption=uploaded_file.name)
            detect_btn = st.button("🔍 Mulai Deteksi", use_container_width=True)

        if detect_btn:
            with col_res:
                st.markdown("<div class='section-title'>✅ Hasil Deteksi</div>",
                            unsafe_allow_html=True)
                with st.spinner("Menganalisis gambar…"):
                    t0 = time.perf_counter()
                    results = model(
                        img_bgr,             # ← kirim BGR ke YOLO
                        imgsz=img_size,
                        conf=0.01,
                        iou=iou_threshold,
                    )
                    elapsed = time.perf_counter() - t0

                # plot() mengembalikan BGR → konversi ke RGB untuk Streamlit
                annotated     = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True,
                         caption=f"Inferensi selesai dalam {elapsed*1000:.0f} ms")

            st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
            render_results(results, model, class_conf_multipliers, conf_threshold)

    else:
        st.markdown("""
        <div class='card' style='text-align:center; padding:3rem 2rem;'>
            <div style='font-size:3rem; margin-bottom:0.8rem;'>☕</div>
            <div style='color:#8a6a40; font-size:1rem;'>
                Upload gambar buah kopi untuk memulai deteksi kematangan.
            </div>
            <div style='color:#5a4020; font-size:0.8rem; margin-top:0.5rem;'>
                Format didukung: JPG, JPEG, PNG, WEBP
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODE: REALTIME KAMERA
# ─────────────────────────────────────────────
elif mode == "🎥 Realtime Kamera":

    st.markdown("<div class='section-title'>🎥 Deteksi Realtime via Kamera</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div style='color:#9a7040; font-size:0.9rem; line-height:1.6;'>
            <strong style='color:#c8a060;'>Petunjuk Penggunaan:</strong><br>
            1. Klik tombol <strong>START</strong> di bawah untuk mengaktifkan kamera.<br>
            2. Arahkan kamera ke tanaman kopi.<br>
            3. Model akan mendeteksi kematangan buah secara langsung.<br>
            4. Atur <em>Confidence</em>, <em>Koreksi per Kelas</em>, dan <em>Ukuran Inferensi</em> di sidebar.
        </div>
    </div>
    """, unsafe_allow_html=True)

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            # Simpan setting sebagai attribute agar bisa diupdate realtime
            self.conf        = conf_threshold
            self.iou         = iou_threshold
            self.imgsz       = img_size
            self.multipliers = class_conf_multipliers.copy()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # frame dari WebRTC sudah BGR — langsung cocok untuk YOLO
            img = frame.to_ndarray(format="bgr24")

            results = model(
                img,
                imgsz=self.imgsz,
                conf=0.01,          # ambil semua kandidat dulu
                iou=self.iou,
                verbose=False,
            )

            # Terapkan class confidence override
            r = results[0]
            if r.boxes is not None and len(r.boxes):
                keep_indices = []
                for i, box in enumerate(r.boxes):
                    cls_id   = int(box.cls[0])
                    raw_conf = float(box.conf[0])
                    label    = model.names[cls_id]
                    mult     = self.multipliers.get(label, 1.0)
                    adj_conf = min(raw_conf * mult, 1.0)
                    if adj_conf >= self.conf:
                        keep_indices.append(i)

                r.boxes = r.boxes[keep_indices] if keep_indices else r.boxes[0:0]

            # plot() output BGR → kembalikan bgr24, warna tetap benar
            annotated = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="kopi-realtime",
        video_processor_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown("""
    <div style='margin-top:1rem; color:#5a4020; font-size:0.78rem; text-align:center;'>
        ⚠️ Performa realtime bergantung pada spesifikasi perangkat dan jaringan.
        Gunakan ukuran inferensi lebih kecil (160/320) untuk kecepatan lebih tinggi.
    </div>
    """, unsafe_allow_html=True)