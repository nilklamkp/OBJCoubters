# Inventory Object Counter (Streamlit)

## โครงสร้าง repo (ต้องอยู่บน GitHub)
```
├─ streamlit_app.py
├─ requirements.txt
└─ models/best.pt     # โมเดลที่เทรนจาก Roboflow dataset (Drink, snacks)
```

## รันในเครื่อง
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy บน Streamlit Community Cloud (share.streamlit.io)
1. สร้าง GitHub repo แล้วอัปโหลดไฟล์ข้างต้น (รวม models/best.pt ขนาดต้องไม่เกิน 100 MB)
2. เข้า https://share.streamlit.io แล้วล็อกอินด้วย GitHub
3. กด Create app → เลือก repo, branch, และ Main file path = streamlit_app.py
4. กด Deploy รอสักครู่จะได้ลิงก์เว็บ

เทรนโมเดล: ใช้ `train.py` จากชุดก่อนหน้า (Google Colab) แล้วนำ
runs/detect/train/weights/best.pt มาวางที่ models/best.pt
