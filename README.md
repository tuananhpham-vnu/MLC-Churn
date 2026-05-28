# Dự Đoán Churn Khách Hàng MLC - Thiết Kế và Phân Tích Thực Nghiệm

**Môn học:** Thiết kế và phân tích thực nghiệm  
**Bộ dữ liệu:** `mlc_churn.csv` - dự đoán khách hàng viễn thông rời mạng  
**Thuật toán sử dụng:** Random Forest  
**Random seed:** `1234`  
**Chỉ số đánh giá chính:** F1-score với lớp dương là `yes`

---

## 1. Tổng quan dự án

Dự án xây dựng và đánh giá mô hình dự đoán khách hàng rời mạng trên bộ dữ liệu `mlc_churn`. Mô hình chính là **Random Forest**, trong đó chỉ thay đổi siêu tham số `max_depth` và giữ các siêu tham số còn lại ở giá trị mặc định.

Các nội dung chính gồm:

- Khám phá và tiền xử lý dữ liệu.
- Phân tích tương quan và ý nghĩa thực tiễn của biến.
- Xây dựng mô hình Random Forest dự đoán churn.
- Đánh giá mô hình bằng F1-score cho lớp `yes`.
- Thiết kế và phân tích thực nghiệm bằng CRD và CRFD.
- Phân tích kết quả bằng mô hình hồi quy tuyến tính `lm()` và báo cáo giá trị trung bình, khoảng tin cậy.

---

## 2. Tóm tắt bộ dữ liệu

**File dữ liệu gốc:** `mlc_churn.csv`

### 2.1. Các nhóm biến chính

- **Địa lý:** `state`, `area_code`
- **Thông tin tài khoản:** `account_length`, `international_plan`, `voice_mail_plan`
- **Thư thoại:** `number_vmail_messages`
- **Sử dụng ban ngày:** `total_day_minutes`, `total_day_calls`, `total_day_charge`
- **Sử dụng buổi tối:** `total_eve_minutes`, `total_eve_calls`, `total_eve_charge`
- **Sử dụng ban đêm:** `total_night_minutes`, `total_night_calls`, `total_night_charge`
- **Sử dụng quốc tế:** `total_intl_minutes`, `total_intl_calls`, `total_intl_charge`
- **Dịch vụ khách hàng:** `number_customer_service_calls`
- **Biến mục tiêu:** `churn` (`yes`/`no`)

### 2.2. Biến mục tiêu

Biến cần dự đoán là `churn`, trong đó:

- `yes`: khách hàng rời mạng, là lớp dương.
- `no`: khách hàng không rời mạng.

Vì bài toán quan tâm đến việc phát hiện khách hàng có khả năng rời mạng, F1-score được tính với **positive class = `yes`**.

---

## 3. Cấu trúc thư mục dự án

```text
2026.DAE.MLC Churn/
├── README.md
├── mlc_churn.csv
│
├── Phase_0_EDA/
│   ├── README.md
│   ├── 00_eda_baseline.py
│   ├── 00_eda_improved.py
│   └── outputs/
│       └── preprocessed_improved.csv
│
├── Phase_1_Model/
│   ├── README.md
│   ├── 01_model_baseline.py
│   ├── 01_model_improved.py
│   └── outputs/
│
├── Phase_2_CRD/
│   ├── README.md
│   ├── 02_crd_baseline.py
│   ├── 02_crd_improved.py
│   ├── 02_crd_analysis.py
│   └── crd_results.csv
│
├── Phase_3_CRFD/
│   ├── README.md
│   ├── 03_crfd_baseline.py
│   ├── 03_crfd_improved.py
│   ├── 03_crfd_analysis.py
│   └── crfd_results.csv
│
├── Phase_4_Report/
│   ├── README.md
│   ├── 04_report_generator.py
│   ├── report_template.md
│   ├── generated_figures/
│   └── MLC_Churn_Report.pdf
│
└── outputs/
    ├── models/
    ├── results/
    └── figures/
```

---

## 4. Giải thích các cách đánh giá mô hình

### 4.1. Holdout

**Holdout** là cách chia dữ liệu một lần thành tập huấn luyện và tập kiểm tra. Ví dụ:

```text
80% dữ liệu -> train
20% dữ liệu -> holdout/test
```

Mô hình học trên tập train và được đánh giá trên tập holdout. F1 trên tập này được gọi là **Holdout F1**. Cách này dễ hiểu nhưng kết quả có thể phụ thuộc vào cách chia dữ liệu.

### 4.2. Cross-validation

**Cross-validation** hay **CV** là cách chia dữ liệu thành nhiều fold. Với k-fold CV, mô hình được huấn luyện và kiểm tra nhiều lần, mỗi lần dùng một fold làm tập kiểm tra và các fold còn lại làm tập huấn luyện.

Ví dụ với 5-fold CV:

```text
Lần 1: Fold 1 test, Fold 2-5 train
Lần 2: Fold 2 test, Fold 1,3,4,5 train
...
Lần 5: Fold 5 test, Fold 1-4 train
```

Kết quả cuối cùng thường được báo cáo bằng trung bình F1 và độ lệch chuẩn hoặc khoảng tin cậy. Trong báo cáo, **CV F1 mean** nên được ưu tiên hơn một lần holdout vì phản ánh hiệu năng ổn định hơn.

---

## 5. Phase 0: EDA và tiền xử lý dữ liệu

**Mục tiêu:** hiểu dữ liệu, kiểm tra chất lượng dữ liệu và chuẩn bị dữ liệu cho mô hình.

Các công việc chính:

1. Đọc dữ liệu `mlc_churn.csv`.
2. Kiểm tra số dòng, số cột, kiểu dữ liệu.
3. Kiểm tra giá trị thiếu.
4. Phân tích phân bố biến mục tiêu `churn`.
5. Phân tích tương quan giữa các biến định lượng.
6. Mã hóa biến phân loại.
7. Lưu dữ liệu đã tiền xử lý vào:

```text
Phase_0_EDA/outputs/preprocessed_improved.csv
```

**File chính:**

- `Phase_0_EDA/00_eda_baseline.py`
- `Phase_0_EDA/00_eda_improved.py`

---

## 6. Phase 1: Xây dựng mô hình Random Forest

**Mục tiêu:** xây dựng mô hình dự đoán churn bằng Random Forest và đánh giá hiệu năng ban đầu.

### 6.1. Yêu cầu chính

- Sử dụng thuật toán Random Forest.
- Chỉ thay đổi siêu tham số `max_depth`.
- Giữ các siêu tham số khác ở giá trị mặc định.
- Sử dụng `random_state=1234`.
- Đánh giá bằng F1-score với lớp dương là `yes`.
- Phân tích sử dụng hoặc loại bỏ biến dựa trên tương quan và ý nghĩa thực tiễn.

### 6.2. Kết quả baseline/improved hiện tại

Kết quả chạy mô hình hiện tại:

```text
Features used: 66
Train F1: 0.999292
CV F1 mean +/- std: 0.743994 +/- 0.026718
Holdout F1: 0.791667
CV F1 mean (95% CI): 0.743994 [0.710819, 0.777168]
```

Diễn giải:

- F1 trên tập train gần bằng 1, cho thấy mô hình học rất tốt trên dữ liệu huấn luyện.
- CV F1 trung bình khoảng `0.744`, phản ánh hiệu năng tổng quát hóa ổn định hơn.
- Holdout F1 khoảng `0.792`, cao hơn trung bình CV, có thể do tập holdout dễ hơn một số fold trong CV.
- Chênh lệch lớn giữa Train F1 và CV F1 cho thấy mô hình có dấu hiệu **overfitting**.

Vì vậy, không nên chỉ nhìn vào Train F1. Khi chọn mô hình, nên ưu tiên **CV F1 mean** và khoảng tin cậy.

### 6.3. So sánh baseline và improved bằng `lm()`

Khoảng tin cậy 95% cho hệ số hồi quy:

```text
                    2.5 %     97.5 %
(Intercept)    0.71644056 0.77154741
modelimproved -0.03896642 0.03896642
```

Diễn giải:

- `(Intercept)` là F1 trung bình ước lượng của mô hình mốc, thường là baseline.
- Hệ số `modelimproved` biểu diễn chênh lệch F1 giữa improved và baseline.
- Khoảng tin cậy của `modelimproved` là `[-0.03897, 0.03897]`, có chứa 0.

Kết luận: chưa có bằng chứng thống kê rõ ràng để khẳng định mô hình improved tốt hơn baseline. Có thể viết trong báo cáo rằng improved và baseline có hiệu năng tương đương, hoặc sự khác biệt chưa có ý nghĩa thống kê.

### 6.4. So sánh các giá trị `max_depth`

Kết quả thực nghiệm hiện tại:

| max_depth | CV F1 mean | Holdout F1 | Train F1 |
|---:|---:|---:|---:|
| None | 0.728707 | 0.791667 | 1.000000 |
| 10 | 0.592736 | 0.614634 | 0.788009 |
| 7 | 0.296547 | 0.236025 | 0.459864 |
| 5 | 0.027472 | 0.094595 | 0.135091 |
| 3 | 0.000000 | 0.014085 | 0.000000 |

Nhận xét:

- `max_depth=None` đạt CV F1 mean cao nhất, nên là cấu hình tốt nhất theo tiêu chí F1 trung bình trên CV.
- Tuy nhiên, Train F1 = 1.0 cho thấy mô hình có dấu hiệu overfitting.
- `max_depth=10` ít overfit hơn nhưng CV F1 thấp hơn khá nhiều.
- Các giá trị `max_depth=3`, `5`, `7` quá nông, khiến mô hình khó nhận diện lớp `yes`, làm F1 rất thấp.

Kết luận tạm thời:

```text
Cấu hình tốt nhất theo CV F1 mean: max_depth=None
```

Tuy nhiên, khi trình bày trong báo cáo cần nêu rõ rủi ro overfitting và đề xuất hướng cải thiện như xử lý mất cân bằng lớp, thử thêm kỹ thuật regularization hoặc cân nhắc class weight trong các phân tích mở rộng.

**File chính:**

- `Phase_1_Model/01_model_baseline.py`
- `Phase_1_Model/01_model_improved.py`

---

## 7. Phase 2: Thực nghiệm CRD

**CRD** là viết tắt của **Completely Randomized Design**, tức thiết kế ngẫu nhiên hoàn toàn.

### 7.1. Mục tiêu

Đánh giá ảnh hưởng của số fold `k` trong k-fold cross-validation đến F1-score của mô hình.

### 7.2. Thiết kế thực nghiệm

- Yếu tố nghiên cứu: số fold `k`.
- Các mức của yếu tố: `k = 3`, `5`, `10`.
- Số lần lặp: `10`.
- Mô hình: Random Forest.
- Chỉ số đánh giá: F1-score với lớp dương là `yes`.
- Chiến lược chia fold: stratified k-fold để giữ tỷ lệ lớp `yes`/`no` tương đối ổn định giữa các fold.
- Random seed: `1234`.

### 7.3. Phân tích thống kê

Các bước phân tích:

1. Tính F1-score cho từng lần lặp và từng giá trị `k`.
2. Báo cáo trung bình, độ lệch chuẩn và khoảng tin cậy 95%.
3. Dùng Levene test để kiểm tra giả định đồng nhất phương sai.
4. Dùng mô hình hồi quy tuyến tính hoặc ANOVA một yếu tố để kiểm tra ảnh hưởng của `k`.
5. Nếu có khác biệt đáng kể, dùng Tukey HSD để so sánh từng cặp.

Công thức mô hình có thể viết:

```text
F1 ~ k
```

### 7.4. Kết quả cần báo cáo

- Bảng trung bình F1 theo từng giá trị `k`.
- Khoảng tin cậy 95% cho từng giá trị `k`.
- Kết quả Levene test.
- Kết quả ANOVA hoặc `lm()`.
- Kết quả Tukey HSD nếu cần.
- Biểu đồ boxplot hoặc mean plot có CI.

**File chính:**

- `Phase_2_CRD/02_crd_baseline.py`
- `Phase_2_CRD/02_crd_improved.py`
- `Phase_2_CRD/02_crd_analysis.py`
- `Phase_2_CRD/crd_results.csv`

---

## 8. Phase 3: Thực nghiệm CRFD

**CRFD** là viết tắt của **Completely Randomized Factorial Design**, tức thiết kế giai thừa ngẫu nhiên hoàn toàn.

### 8.1. Mục tiêu

Phân tích đồng thời ảnh hưởng của hai yếu tố `k` và `max_depth` đến F1-score, đồng thời kiểm tra xem hai yếu tố này có tương tác với nhau hay không.

### 8.2. Thiết kế thực nghiệm

- Yếu tố A: số fold `k`
  - Các mức: `3`, `5`, `10`
- Yếu tố B: `max_depth`
  - Các mức: `3`, `5`, `None`
- Số lần lặp: `10`
- Tổng số quan sát: `3 × 3 × 10 = 90`
- Chỉ số đánh giá: F1-score với lớp dương là `yes`
- Random seed: `1234`

### 8.3. Phân tích thống kê

Mô hình phân tích:

```text
F1 ~ k + max_depth + k:max_depth
```

Trong đó:

- `k`: hiệu ứng chính của số fold.
- `max_depth`: hiệu ứng chính của độ sâu cây.
- `k:max_depth`: hiệu ứng tương tác giữa số fold và độ sâu cây.

Các nội dung cần phân tích:

1. Ảnh hưởng chính của `k` đến F1.
2. Ảnh hưởng chính của `max_depth` đến F1.
3. Tương tác giữa `k` và `max_depth`.
4. Trung bình và khoảng tin cậy cho từng nhóm.
5. Cấu hình có F1 trung bình tốt nhất.

### 8.4. Kết quả cần báo cáo

- Bảng F1 trung bình theo từng giá trị `k`.
- Bảng F1 trung bình theo từng giá trị `max_depth`.
- Bảng F1 trung bình cho từng tổ hợp `k × max_depth`.
- Kết quả `lm()` hoặc ANOVA hai yếu tố.
- Biểu đồ hiệu ứng chính.
- Biểu đồ tương tác.
- Nhận xét về ý nghĩa thống kê và ý nghĩa thực tế.

**File chính:**

- `Phase_3_CRFD/03_crfd_baseline.py`
- `Phase_3_CRFD/03_crfd_improved.py`
- `Phase_3_CRFD/03_crfd_analysis.py`
- `Phase_3_CRFD/crfd_results.csv`

---

## 9. Phase 4: Viết báo cáo

**Mục tiêu:** tổng hợp phương pháp, thiết kế thực nghiệm, kết quả và thảo luận thành báo cáo cuối cùng.

### 9.1. Cấu trúc báo cáo đề xuất

1. **Giới thiệu**
   - Bối cảnh bài toán churn prediction.
   - Mục tiêu nghiên cứu.
   - Mô tả ngắn về dữ liệu.

2. **Dữ liệu và tiền xử lý**
   - Mô tả biến.
   - Kiểm tra thiếu dữ liệu.
   - Phân bố lớp `churn`.
   - Mã hóa biến phân loại.
   - Phân tích tương quan và lý do giữ/loại biến.

3. **Phương pháp mô hình hóa**
   - Random Forest.
   - Siêu tham số `max_depth`.
   - F1-score cho lớp dương `yes`.
   - Holdout và cross-validation.

4. **Thiết kế thực nghiệm**
   - CRD với yếu tố `k`.
   - CRFD với hai yếu tố `k` và `max_depth`.
   - Random seed và số lần lặp.

5. **Kết quả và phân tích**
   - Kết quả mô hình ban đầu.
   - So sánh các giá trị `max_depth`.
   - Kết quả CRD.
   - Kết quả CRFD.
   - Kết quả `lm()`, khoảng tin cậy và kiểm định thống kê.

6. **Thảo luận**
   - Vì sao `max_depth` nhỏ cho F1 thấp.
   - Dấu hiệu overfitting khi `max_depth=None`.
   - Ý nghĩa thực tế của việc dự đoán lớp `yes`.
   - Hạn chế của mô hình và thực nghiệm.

7. **Kết luận**
   - Tóm tắt kết quả chính.
   - Cấu hình tốt nhất theo CV F1.
   - Hướng cải thiện tiếp theo.

### 9.2. Kết quả giao nộp

- Báo cáo PDF cuối cùng.
- File kết quả CRD: `crd_results.csv`.
- File kết quả CRFD: `crfd_results.csv`.
- Mã nguồn tái lập kết quả.

---

## 10. Các tham số quan trọng

| Tham số | Giá trị |
|---|---|
| Random seed | `1234` |
| Thuật toán | Random Forest |
| Siêu tham số thay đổi | `max_depth` |
| Các siêu tham số khác | Giữ mặc định |
| Chỉ số đánh giá | F1-score |
| Lớp dương | `yes` |
| Chiến lược chia fold | Stratified k-fold |
| Giá trị `k` trong CRD | `3`, `5`, `10` |
| Số lần lặp CRD | `10` |
| Giá trị `max_depth` trong CRFD | `3`, `5`, `None` |
| Số lần lặp CRFD | `10` |

---

## 11. Thứ tự chạy chương trình

Chạy từ thư mục gốc của dự án:

```powershell
cd "D:\Folder F\phamtuananh@23020010\UET.iSEML\2026.DAE.MLC Churn"
```

Sau đó chạy lần lượt:

```bash
# Phase 0: EDA và tiền xử lý
python Phase_0_EDA/00_eda_baseline.py
python Phase_0_EDA/00_eda_improved.py

# Phase 1: Xây dựng và đánh giá mô hình
python Phase_1_Model/01_model_baseline.py
python Phase_1_Model/01_model_improved.py

# Phase 2: Thực nghiệm CRD
python Phase_2_CRD/02_crd_baseline.py
python Phase_2_CRD/02_crd_analysis.py

# Phase 3: Thực nghiệm CRFD
python Phase_3_CRFD/03_crfd_baseline.py
python Phase_3_CRFD/03_crfd_analysis.py

# Phase 4: Tạo báo cáo
python Phase_4_Report/04_report_generator.py
```

Nếu đang đứng trực tiếp trong `Phase_1_Model`, có thể chạy:

```powershell
python .\01_model_baseline.py
python .\01_model_improved.py
```

---

## 12. Câu diễn giải có thể dùng trong báo cáo

### 12.1. Về hiệu năng mô hình

Mô hình Random Forest đạt F1 trung bình qua cross-validation khoảng `0.744` với khoảng tin cậy 95% là `[0.711, 0.777]`. F1 trên tập holdout đạt khoảng `0.792`. Tuy nhiên, F1 trên tập huấn luyện gần bằng 1, cho thấy mô hình có xu hướng overfitting. Vì vậy, kết quả cross-validation được ưu tiên sử dụng để đánh giá khả năng tổng quát hóa.

### 12.2. Về so sánh baseline và improved

Khoảng tin cậy 95% của hệ số `modelimproved` là `[-0.039, 0.039]`, có chứa 0. Do đó, chưa có bằng chứng thống kê để khẳng định phiên bản improved tạo ra cải thiện F1 rõ rệt so với baseline. Hai phiên bản có thể được xem là có hiệu năng tương đương trong phạm vi thực nghiệm hiện tại.

### 12.3. Về lựa chọn `max_depth`

Trong các giá trị `max_depth` được so sánh, `max_depth=None` đạt CV F1 trung bình cao nhất. Tuy nhiên, Train F1 bằng 1.0 cho thấy mô hình có dấu hiệu overfitting. Các giá trị `max_depth` nhỏ như 3 hoặc 5 làm mô hình quá đơn giản, gần như không nhận diện được lớp churn `yes`, dẫn đến F1 rất thấp.

---

## 13. Danh sách kiểm tra trước khi nộp

- [ ] Đã chạy EDA và lưu dữ liệu tiền xử lý.
- [ ] Đã phân tích tương quan và giải thích việc giữ/loại biến.
- [ ] Đã xây dựng mô hình Random Forest với `random_state=1234`.
- [ ] Đã đánh giá bằng F1-score với lớp dương `yes`.
- [ ] Đã so sánh các giá trị `max_depth`.
- [ ] Đã thực hiện CRD với các giá trị `k = 3, 5, 10`.
- [ ] Đã thực hiện Levene test và Tukey HSD khi cần.
- [ ] Đã thực hiện CRFD với các yếu tố `k` và `max_depth`.
- [ ] Đã phân tích hiệu ứng chính và tương tác.
- [ ] Đã báo cáo trung bình và khoảng tin cậy 95% trong các so sánh.
- [ ] Đã tạo biểu đồ cần thiết.
- [ ] Đã tạo báo cáo PDF cuối cùng.
- [ ] Tất cả kết quả có thể tái lập với seed `1234`.

---

## 14. Ghi chú quan trọng

- Không nên kết luận mô hình improved tốt hơn baseline nếu khoảng tin cậy của hệ số so sánh còn chứa 0.
- Không nên chọn mô hình chỉ dựa trên Train F1 vì Train F1 cao có thể là dấu hiệu overfitting.
- Với dữ liệu mất cân bằng lớp, F1-score của lớp `yes` có ý nghĩa hơn accuracy.
- Khi `max_depth` quá nhỏ, mô hình có thể dự đoán rất ít hoặc không dự đoán lớp `yes`, khiến F1 gần 0.
- Trong báo cáo, cần phân biệt rõ **ý nghĩa thống kê** và **ý nghĩa thực tế**.

---

**Cập nhật lần cuối:** 2026-05-27  
**Trạng thái:** Đã cập nhật theo kết quả Phase 1 và yêu cầu phân tích thực nghiệm.
