# Phase 0: Khám Phá Dữ Liệu & Xử Lý Trước

**Mục tiêu:** Hiểu cấu trúc bộ dữ liệu, xác định các vấn đề chất lượng dữ liệu, và chuẩn bị dữ liệu cho mô hình hóa

**Điểm:** Bắt buộc (không được chấm điểm nhưng quan trọng)

---

## Công Việc

### 1. Tải & Tổng Quan Dữ Liệu
- Tải file CSV
- Hiển thị hình dạng, tên cột, kiểu dữ liệu
- Hiển thị hàng đầu/cuối
- In thống kê bộ dữ liệu

### 2. Phân Tích Giá Trị Thiếu
- Xác định các giá trị thiếu theo cột
- Trực quan hóa mẫu thiếu
- Quyết định chiến lược xử lý

### 3. Kiểu Đặc Trưng & Phân Phối
- Xác định các đặc trưng phân loại vs số
- Phân tích phân phối biến mục tiêu (tỷ lệ churn)
- Kiểm tra mất cân bằng lớp
- Trực quan hóa phân phối cho các đặc trưng số

### 4. Phân Tích Tương Quan
- Tính ma trận tương quan cho các đặc trưng số
- Trực quan hóa bản đồ nhiệt
- Xác định các cặp đặc trưng tương quan cao
- Gắn cờ các đặc trưng dư thừa tiềm ẩn

### 5. Cân Nhắc Kỹ Thuật Đặc Trưng
- Mã hóa các biến phân loại (state, area_code, plans)
- Xử lý area_code (nhiều danh mục)
- Xem xét tương tác hoặc chuyển đổi
- Chuẩn bị bộ dữ liệu cho mô hình hóa

### 6. Chiến Lược Chia Dữ Liệu
- Lập kế hoạch chia train/test hoặc cách tiếp cận xác thực chéo
- Đảm bảo phân tầng cho mất cân bằng lớp
- Ghi lại việc sử dụng random seed (seed=1234)

---

## Cách Tiếp Cận Cơ Bản (`00_eda_baseline.py`)

```python
# 1. Tải dữ liệu
# 2. Basic info() và describe()
# 3. Kiểm tra giá trị thiếu
# 4. In phân phối churn
# 5. Hiển thị tương quan cho các cột số
# 6. Mã hóa đặc trưng cơ bản cho các biến phân loại
```

**Đầu ra:**
- Thống kê tóm tắt bàn phím lệnh
- Ma trận tương quan (các đặc trưng số)
- Phần trăm phân phối churn

---

## Cách Tiếp Cận Cải Tiến (`00_eda_improved.py`)

Cải thiện so với cơ bản:
1. **Trực quan hóa Toàn diện**
   - Phân phối churn (biểu đồ thanh)
   - Phân phối đặc trưng theo trạng thái churn (subplots)
   - Bản đồ nhiệt tương quan với phong cách tốt hơn
   - Trực quan hóa giá trị thiếu

2. **Tóm Tắt Thống Kê**
   - Thống kê tóm tắt theo nhóm churn
   - So sánh đặc trưng (rời mạng vs không rời mạng)
   - Kiểm tra thống kê ý nghĩa đặc trưng

3. **Quy Trình Xử Lý Trước Dữ Liệu**
   - Xử lý mã hóa phân loại một cách có hệ thống
   - Cân nhắc về chuẩn hóa đặc trưng
   - Ghi lại các bước xử lý trước để tái tạo

4. **Phân Tích Khám Phá**
   - Tầm quan trọng đặc trưng (tương quan với mục tiêu)
   - Phát hiện ngoại lệ
   - Báo cáo chất lượng dữ liệu

---

## Kết Quả Dự Kiến

### Đặc Điểm Dữ Liệu
- **Mất Cân Bằng Lớp:** Kỳ vọng ~85% churn=no, ~15% churn=yes
- **Đặc trưng:** ~20 tổng cộng các đặc trưng (số + phân loại)
- **Khách hàng:** ~3,333 bản ghi

### Các Đặc Trưng Cần Cân Nhắc
✓ **Giữ:** Các đặc trưng có mối quan hệ rõ ràng với churn
- Cuộc gọi dịch vụ khách hàng (số cao cho thấy sự không hài lòng)
- Mô hình sử dụng (tổng phút theo giai đoạn)
- Độ dài tài khoản
- Thông tin kế hoạch

✗ **Loại bỏ/Kết hợp:** Các đặc trưng dư thừa
- Các biến phí (bắt nguồn từ phút)
- Mã khu vực (quá nhiều danh mục, giá trị dự đoán thấp)
- Một số thông tin trạng thái (biến thiên địa lý tối thiểu)

---

## Các Quyết Định Chính

1. **Chiến Lược Mã Hóa Phân Loại:**
   - One-hot mã hóa: international_plan, voice_mail_plan, churn
   - Xử lý area_code: Nhóm danh mục hiếm hoặc loại bỏ
   - Xử lý state: Sử dụng one-hot hoặc loại bỏ

2. **Lựa Chọn Đặc Trưng:**
   - Loại bỏ các cột phí nếu có phút
   - Loại bỏ các cặp đặc trưng có tương quan cao
   - Giữ các đặc trưng có thể giải thích được

3. **Phân Tầng:**
   - Luôn sử dụng chia phân tầng/fold (duy trì tỷ lệ churn)
   - Theo dõi random seed = 1234

---

## File Đầu Ra

```
Phase_0_EDA/
├── README.md (file này)
├── 00_eda_baseline.py
├── 00_eda_improved.py
└── outputs/
    ├── correlation_matrix.csv
    ├── feature_summary.txt
    ├── churn_distribution.png
    └── correlation_heatmap.png
```

---

## Tiêu Chí Thành Công

✓ Hiểu tất cả các đặc trưng và ý nghĩa của chúng  
✓ Xác định tỷ lệ churn và mất cân bằng lớp  
✓ Tạo ma trận tương quan và xác định các đặc trưng dư thừa  
✓ Lập kế hoạch chiến lược kỹ thuật đặc trưng và mã hóa  
✓ Chuẩn bị bộ dữ liệu sạch cho mô hình hóa (Phase 1)  
✓ Ghi lại tất cả các quyết định xử lý trước  

---

## Hướng Dẫn Chạy

```bash
# Khám phá cơ bản
python 00_eda_baseline.py

# Khám phá nâng cao với trực quan hóa
python 00_eda_improved.py
```

---

## Ghi Chú

- Sử dụng seed=1234 cho bất kỳ ngẫu nhiên hóa nào
- Luôn bảo tồn phân tầng (duy trì tỷ lệ churn)
- Ghi lại tất cả các quyết định để tái tạo
- Giữ xử lý trước nhất quán trên tất cả các giai đoạn
- Xuất dữ liệu sạch cho Phase 1
