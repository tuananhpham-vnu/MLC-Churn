# Dự Đoán Churn Khách Hàng MLC - Thiết Kế và Phân Tích Thực Nghiệm

**Môn học:** Thiết kế và phân tích thực nghiệm  
**Bộ dữ liệu:** mlc_churn - Dự đoán rời mạng khách hàng viễn thông  
**Tổng điểm:** 10 điểm

---

## Tổng Quan Dự Án

Dự án này thực hiện một nghiên cứu máy học toàn diện về dự đoán khách hàng rời mạng sử dụng thuật toán Random Forest. Công việc bao gồm:
- Xây dựng mô hình dự đoán
- Thực hiện hai thiết kế thực nghiệm (CRD và CRFD)
- Phân tích thống kê kết quả
- Viết báo cáo chi tiết

**Ngôn ngữ lập trình:** Python 3.x  
**Random Seed:** 1234 (tất cả thực nghiệm)  
**Chỉ số đánh giá:** F1 Score (lớp dương = "yes")

---

## Tóm Tắt Bộ Dữ Liệu

**File:** `mlc_churn.csv`

### Các Đặc Trưng:
- **Địa lý:** state, area_code
- **Thông tin tài khoản:** account_length, international_plan, voice_mail_plan
- **Thư thoại di động:** number_vmail_messages
- **Sử dụng ban ngày:** total_day_minutes, total_day_calls, total_day_charge
- **Sử dụng ban tối:** total_eve_minutes, total_eve_calls, total_eve_charge
- **Sử dụng ban đêm:** total_night_minutes, total_night_calls, total_night_charge
- **Sử dụng quốc tế:** total_intl_minutes, total_intl_calls, total_intl_charge
- **Dịch vụ khách hàng:** number_customer_service_calls
- **Biến mục tiêu:** churn (yes/no)

---

## Cấu Trúc Dự Án

```
2026.DAE.MLC Churn/
├── README.md                           # File này - Kế hoạch dự án chính
├── mlc_churn.csv                       # Bộ dữ liệu
│
├── Phase_0_EDA/                        # Khám phá dữ liệu & Xử lý trước
│   ├── README.md                       # Hướng dẫn Phase 0
│   ├── 00_eda_baseline.py              # EDA cơ bản
│   └── 00_eda_improved.py              # EDA nâng cao với biểu đồ
│
├── Phase_1_Model/                      # Mô hình Random Forest cơ sở
│   ├── README.md                       # Hướng dẫn Phase 1
│   ├── 01_model_baseline.py            # Xây dựng mô hình RF cơ bản với phân tích tương quan
│   └── 01_model_improved.py            # Tối ưu hóa mô hình và chọn lọc đặc trưng
│
├── Phase_2_CRD/                        # Thực nghiệm CRD (Thiết kế ngẫu nhiên hoàn toàn)
│   ├── README.md                       # Hướng dẫn Phase 2
│   ├── 02_crd_baseline.py              # CRD cơ bản với k-fold lặp lại
│   ├── 02_crd_improved.py              # CRD nâng cao với xác thực
│   ├── 02_crd_analysis.py              # Phân tích thống kê (leveneTest, TukeyHSD)
│   └── crd_results.csv                 # Kết quả: Kết quả thực nghiệm CRD
│
├── Phase_3_CRFD/                       # Thực nghiệm CRFD (Thiết kế giai thừa ngẫu nhiên hoàn toàn)
│   ├── README.md                       # Hướng dẫn Phase 3
│   ├── 03_crfd_baseline.py             # CRFD cơ bản với hai yếu tố
│   ├── 03_crfd_improved.py             # CRFD nâng cao với xác thực
│   ├── 03_crfd_analysis.py             # Phân tích thống kê (hiệu ứng tương tác)
│   └── crfd_results.csv                # Kết quả: Kết quả thực nghiệm CRFD
│
├── Phase_4_Report/                     # Viết Báo Cáo
│   ├── README.md                       # Hướng dẫn Phase 4
│   ├── 04_report_generator.py          # Tạo biểu đồ báo cáo và thống kê
│   ├── report_template.md              # Mẫu báo cáo markdown
│   ├── generated_figures/              # Kết quả: Biểu đồ phân tích
│   └── MLC_Churn_Report.pdf            # Báo cáo PDF cuối cùng (sẽ tạo)
│
└── outputs/
    ├── models/                         # File mô hình lưu trữ
    ├── results/                        # File CSV kết quả
    └── figures/                        # Biểu đồ được tạo
```

---

## Phân Tích Từng Giai Đoạn

### **Phase 0: Khám Phá Dữ Liệu & Xử Lý Trước** (Bắt buộc)
**Mục tiêu:** Hiểu cấu trúc dữ liệu, xử lý giá trị thiếu, chuẩn bị cho mô hình hóa

**Kết quả giao hàng:**
- Tóm tắt và thống kê dữ liệu
- Phân tích giá trị thiếu
- Tương quan đặc trưng
- Phân bố lớp (cân bằng churn)
- Quy trình xử lý trước dữ liệu

**File:**
- `Phase_0_EDA/00_eda_baseline.py` - Khám phá cơ bản
- `Phase_0_EDA/00_eda_improved.py` - Trực quan hóa nâng cao

---

### **Phase 1: Mô Hình Random Forest Cơ Sở** (2 điểm)
**Mục tiêu:** Xây dựng mô hình dự đoán ban đầu và phân tích tầm quan trọng của đặc trưng

**Yêu cầu:**
- Sử dụng thuật toán Random Forest
- Phân tích tương quan giữa các đặc trưng
- Loại bỏ các đặc trưng không liên quan dựa trên tương quan & ý nghĩa thực tế
- Chỉ sử dụng `max_depth` là siêu tham số (các cái khác mặc định)
- Random seed = 1234
- Đánh giá với F1 score (lớp dương = "yes")

**Cách tiếp cận cơ bản:**
1. Tải và xử lý trước dữ liệu
2. Tính toán tương quan đặc trưng
3. Loại bỏ các đặc trưng tương quan cao
4. Huấn luyện RF với siêu tham số mặc định
5. Đánh giá hiệu năng mô hình

**Cải tiến:**
1. Phân tích tầm quan trọng của đặc trưng
2. Các ngưỡng tương quan khác nhau
3. Xác thực chéo mô hình
4. Khám phá mở rộng tính năng

**File:**
- `Phase_1_Model/01_model_baseline.py` - Mô hình cơ bản
- `Phase_1_Model/01_model_improved.py` - Phiên bản cải tiến

---

### **Phase 2: Thực Nghiệm CRD** (3 điểm)
**Mục tiêu:** Thiết kế và phân tích thực nghiệm hoàn toàn ngẫu nhiên với các giá trị k-fold khác nhau

**Thiết kế thực nghiệm:**
- Phương pháp: Xác thực chéo k-fold lặp lại
- Giá trị k: 3, 5, 10
- Lần lặp lại: 10
- Chiến lược fold: Phân tầng (xử lý mất cân bằng lớp)
- Random seed: 1234

**Yêu cầu:**

1. **So sánh Phương sai (1 điểm)**
   - Sử dụng test Levene để so sánh phương sai F1 trên các giá trị k khác nhau
   - Báo cáo thống kê kiểm tra và giá trị p
   - Kết luận về tính đồng nhất của phương sai

2. **Đánh giá Tác động của k (2 điểm)**
   - Phân tích tác động của k đến hiệu suất mô hình
   - Thực hiện kiểm tra hậu hoc Tukey HSD để so sánh từng cặp
   - Tạo biểu đồ so sánh (3 nhóm)
   - Báo cáo giá trị trung bình và khoảng tin cậy

**Cách tiếp cận cơ bản:**
1. Thiết lập k-fold lặp lại với k={3,5,10}, lần lặp=10
2. Huấn luyện mô hình RF trên mỗi kết hợp fold
3. Thu thập điểm số F1
4. Thực hiện kiểm tra Levene
5. Tiến hành ANOVA một chiều và Tukey HSD
6. Tạo biểu đồ so sánh

**Cải tiến:**
1. Xác minh k-fold phân tầng
2. So sánh kiểm tra thống kê đa lần
3. Phân tích lỗi theo cấu hình fold
4. Biểu đồ hộp và biểu đồ violin
5. Phân tích quyền lực thống kê

**File:**
- `Phase_2_CRD/02_crd_baseline.py` - Thực nghiệm CRD cơ bản
- `Phase_2_CRD/02_crd_improved.py` - Xác thực nâng cao
- `Phase_2_CRD/02_crd_analysis.py` - Phân tích thống kê
- Kết quả: `Phase_2_CRD/crd_results.csv`

---

### **Phase 3: Thực Nghiệm CRFD** (4 điểm)
**Mục tiêu:** Phân tích tương tác giữa k và max_depth bằng thiết kế giai thừa

**Thiết kế thực nghiệm:**
- Yếu tố: 
  - Yếu tố A (k): 3, 5, 10
  - Yếu tố B (max_depth): 3, 5, Không giới hạn
- Lần lặp lại: 10
- Chiến lược fold: Phân tầng
- Random seed: 1234
- Tổng kết hợp: 3 × 3 × 10 = 90 thực nghiệm

**Yêu cầu:**

1. **Phân tích Hiệu ứng Chính (3 điểm)**
   - Đánh giá tác động của k đến hiệu suất mô hình (với trung bình và CI)
   - Đánh giá tác động của max_depth đến hiệu suất mô hình (với trung bình và CI)
   - Tạo biểu đồ cho thấy hiệu ứng chính
   - Kiểm tra ý nghĩa thống kê

2. **Phân tích Tương tác (1 điểm)**
   - Kiểm tra ý nghĩa thống kê của tương tác k × max_depth
   - Trực quan hóa biểu đồ tương tác
   - Mô tả ảnh hưởng tương tác đến hiệu suất mô hình (nếu có)
   - Phân tích tác động thực tế

**Cách tiếp cận cơ bản:**
1. Thiết lập thiết kế giai thừa 2 yếu tố với tất cả kết hợp
2. Huấn luyện mô hình RF cho mỗi kết hợp (với k-fold phân tầng, lặp=10)
3. Thu thập điểm số F1 (9 × 10 = 90 quan sát)
4. Thực hiện ANOVA hai chiều
5. Phân tích hiệu ứng chính và tương tác

**Cải tiến:**
1. Ước tính kích thước hiệu ứng
2. Biểu đồ chẩn đoán mô hình (phần dư, QQ-plot)
3. Phương pháp so sánh đa lần
4. Kiểm tra độ bền
5. Gợi ý cấu hình tốt nhất

**File:**
- `Phase_3_CRFD/03_crfd_baseline.py` - Thực nghiệm CRFD cơ bản
- `Phase_3_CRFD/03_crfd_improved.py` - Xác thực nâng cao
- `Phase_3_CRFD/03_crfd_analysis.py` - Phân tích thống kê
- Kết quả: `Phase_3_CRFD/crfd_results.csv`

---

### **Phase 4: Viết Báo Cáo** (1 điểm)
**Mục tiêu:** Tổng hợp tất cả các phát hiện thành báo cáo PDF toàn diện

**Cấu trúc báo cáo:**
1. **Giới thiệu**
   - Bối cảnh vấn đề và động lực
   - Mô tả bộ dữ liệu
   - Câu hỏi nghiên cứu

2. **Phương pháp**
   - Tổng quan thuật toán Random Forest
   - Thiết kế thực nghiệm (CRD và CRFD)
   - Phương pháp phân tích thống kê

3. **Kết quả**
   - Hiệu suất mô hình cơ sở
   - Kết quả thực nghiệm CRD
   - Kết quả thực nghiệm CRFD
   - Kiểm tra thống kê và trực quan hóa

4. **Thảo luận**
   - Giải thích các phát hiện chính
   - Ứng dụng thực tế
   - Hạn chế

5. **Kết luận**
   - Tóm tắt
   - Khuyến nghị

**Kết quả giao hàng:**
- Báo cáo PDF (tối đa 10-15 trang)
- 2 file CSV với kết quả thực nghiệm (CRD và CRFD)

**File:**
- `Phase_4_Report/04_report_generator.py` - Tạo biểu đồ và thống kê
- `Phase_4_Report/report_template.md` - Mẫu báo cáo
- Kết quả: `Phase_4_Report/MLC_Churn_Report.pdf`

---

## Các Tham Số & Yêu Cầu Chính

| Tham Số | Giá Trị |
|---------|---------|
| Random Seed | 1234 |
| Chỉ số Đánh giá | F1 Score (lớp dương = "yes") |
| Lớp Dương | "yes" (rời mạng) |
| Chiến Lược Fold | Phân tầng |
| Giá trị k của CRD | 3, 5, 10 |
| Lần Lặp CRD | 10 |
| Giá trị max_depth của CRFD | 3, 5, Không giới hạn |
| Lần Lặp CRFD | 10 |
| Siêu tham số RF | Chỉ thay đổi max_depth, các cái khác mặc định |

---

## Chiến Lược Thực Hiện

### Mô Hình Cơ Bản → Cải Tiến

Mỗi giai đoạn tuân theo mô hình này:
1. **Script Cơ Bản** (`*_baseline.py`)
   - Triển khai tối thiểu, sạch sẽ
   - Tập trung vào các yêu cầu cốt lõi
   - Đầu ra và ghi nhật ký rõ ràng

2. **Script Cải Tiến** (`*_improved.py`)
   - Xác thực và xử lý lỗi nâng cao
   - Kiểm tra độ bền bổ sung
   - Trực quan hóa và báo cáo tốt hơn

3. **Script Phân Tích** (`*_analysis.py`)
   - Kiểm tra thống kê và giải thích
   - Trực quan hóa toàn diện
   - Tài liệu kết quả chi tiết

---

## Thư Viện Bắt Buộc

```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.13.0
```

---

## Thứ Tự Thực Hiện

1. **Phase 0:** Chạy EDA để hiểu dữ liệu
2. **Phase 1:** Xây dựng mô hình cơ sở
3. **Phase 2:** Thực hiện thực nghiệm CRD
4. **Phase 3:** Thực hiện thực nghiệm CRFD
5. **Phase 4:** Tạo báo cáo

---

## Danh Sách Kiểm Tra Gửi

- [ ] Phase 0: EDA hoàn tất và dữ liệu được xác thực
- [ ] Phase 1: Mô hình RF cơ sở với phân tích đặc trưng
- [ ] Phase 2: Thực nghiệm CRD với kiểm tra Levene và Tukey HSD
- [ ] Phase 2: CSV kết quả CRD đã lưu
- [ ] Phase 3: Thực nghiệm CRFD với hiệu ứng chính và tương tác
- [ ] Phase 3: CSV kết quả CRFD đã lưu
- [ ] Phase 4: Báo cáo PDF đã tạo
- [ ] Tất cả mã chạy không có lỗi
- [ ] Kết quả có thể tái tạo với seed=1234

---

## Bắt Đầu Nhanh

```bash
# Phase 0: Khám phá dữ liệu
python Phase_0_EDA/00_eda_baseline.py

# Phase 1: Xây dựng mô hình
python Phase_1_Model/01_model_baseline.py

# Phase 2: Thực nghiệm CRD
python Phase_2_CRD/02_crd_baseline.py
python Phase_2_CRD/02_crd_analysis.py

# Phase 3: Thực nghiệm CRFD
python Phase_3_CRFD/03_crfd_baseline.py
python Phase_3_CRFD/03_crfd_analysis.py

# Phase 4: Tạo báo cáo
python Phase_4_Report/04_report_generator.py
```

---

## Ghi Chú

- Tất cả random seed phải được đặt thành **1234**
- Sử dụng **k-fold phân tầng** để xử lý mất cân bằng lớp
- Báo cáo **giá trị trung bình và khoảng tin cậy** trong tất cả so sánh
- Bao gồm cả **ý nghĩa thống kê** và **ý nghĩa thực tế**
- Tạo tất cả các biểu đồ bắt buộc với nhãn và chú thích rõ ràng
- Lưu kết quả trung gian để tái tạo

---

**Cập nhật lần cuối:** 2026-05-23  
**Trạng thái:** Kế hoạch dự án sẵn sàng - Sẵn sàng bắt đầu Phase 0
#   M L C - C h u r n  
 