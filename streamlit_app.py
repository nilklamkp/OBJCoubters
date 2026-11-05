import os
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Inventory Object Counter", page_icon="📦", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")


@st.cache_resource
def load_model():
    # ถ้ายังไม่มี best.pt จะใช้ yolov8n.pt (คลาส COCO) เพื่อทดสอบเท่านั้น
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH), True
    return YOLO("yolov8n.pt"), False


model, custom = load_model()

st.title("📦 Inventory Object Counter")
st.caption("อัปโหลดรูปชั้นวางสินค้า ระบบจะตรวจจับและนับจำนวนให้")

if not custom:
    st.warning("ยังไม่พบ models/best.pt ตอนนี้ใช้ yolov8n.pt ทดสอบ (ไม่รู้จัก Drink / snacks)")

with st.sidebar:
    st.header("ตั้งค่า")
    conf = st.slider("ความมั่นใจขั้นต่ำ (confidence)", 0.05, 0.90, 0.25, 0.05)
    iou = st.slider("IoU (ตัดกรอบซ้อน)", 0.1, 0.9, 0.45, 0.05)

files = st.file_uploader(
    "เลือกรูปภาพ (เลือกได้หลายรูป)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if files and st.button("เริ่มนับ", type="primary"):
    total = Counter()
    rows, shown = [], []
    bar = st.progress(0.0, text="กำลังประมวลผล...")

    for i, f in enumerate(files):
        img = np.array(Image.open(f).convert("RGB"))
        r = model.predict(img, conf=conf, iou=iou, verbose=False)[0]
        counts = Counter(model.names[int(c)] for c in r.boxes.cls.tolist())
        total.update(counts)
        annotated = r.plot()[..., ::-1]  # BGR -> RGB
        shown.append((f.name, annotated, sum(counts.values())))
        rows.append({"ไฟล์": f.name, "รวม": sum(counts.values()), **counts})
        bar.progress((i + 1) / len(files), text=f"ประมวลผลแล้ว {i + 1}/{len(files)}")
    bar.empty()

    st.subheader("สรุปผล")
    cols = st.columns(2 + len(total))
    cols[0].metric("จำนวนรวมทั้งหมด", sum(total.values()))
    cols[1].metric("จำนวนรูป", len(files))
    for c, (name, n) in zip(cols[2:], total.items()):
        c.metric(name, n)

    df = pd.DataFrame(rows).fillna(0)
    for c in df.columns[1:]:
        df[c] = df[c].astype(int)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "ดาวน์โหลด CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "count_summary.csv",
        "text/csv",
    )

    if len(total) > 0:
        st.bar_chart(pd.Series(dict(total)))

    st.subheader("ผลการตรวจจับ")
    grid = st.columns(3)
    for idx, (name, im, n) in enumerate(shown):
        grid[idx % 3].image(im, caption=f"{name} — {n} ชิ้น", use_container_width=True)
elif not files:
    st.info("อัปโหลดรูปเพื่อเริ่มต้น")
