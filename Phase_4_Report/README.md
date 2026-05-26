# Phase 4: Viết Báo Cáo

**Mục tiêu:** Viết báo cáo PDF toàn diện tổng hợp tất cả các phát hiện

**Điểm:** 1 điểm

---

## Tổng Quan Báo Cáo

### Mục Đích
Ghi lại hoàn chỉnh nghiên cứu máy học về dự đoán churn khách hàng, bao gồm:
- Quy trình phát triển mô hình
- Kết quả thực nghiệm thiết kế
- Phân tích thống kê các phát hiện
- Kết luận và khuyến nghị

### Định Dạng
- Tài liệu PDF
- Độ dài: 10-15 trang (được khuyến nghị)
- Cấu trúc chuyên nghiệp với các phần, biểu đồ và bảng
- Viết rõ ràng bằng Tiếng Anh hoặc Tiếng Việt (chỉ định)

### Đối Tượng
- Học Thuật: Giáo viên hướng dẫn và các bạn cùng lớp
- Kỹ Thuật: Người đọc quen thuộc với ML và thống kê
- Người Ra Quyết Định: Muốn có hiểu biết thực tế

---

## Cấu Trúc Báo Cáo

### 1. Trang Tiêu Đề
- Tiêu đề: "Dự Đoán Churn Khách Hàng: Thiết Kế và Phân Tích Thực Nghiệm"
- Tên tác giả và mã sinh viên
- Tên môn học
- Ngày
- Tổ Chức

---

### 2. Tóm Tắt (½ trang)
**Nội Dung:**
- Tóm tắt ngắn gọn về vấn đề
- Tổng quan phương pháp
- Các phát hiện chính
- Kết luận chính

**Ví Dụ:**
> Nghiên cứu này điều tra dự đoán churn khách hàng sử dụng phân loại Random Forest trên bộ dữ liệu MLC viễn thông. Chúng tôi thiết kế hai thực nghiệm sử dụng thiết kế hoàn toàn ngẫu nhiên (CRD và CRFD) để phân tích cách k-fold xác thực chéo và siêu tham số max_depth của Random Forest ảnh hưởng đến hiệu suất mô hình (điểm F1). Kết quả cho thấy [KẾT QUẢ CHÍNH 1] và [KẾT QUẢ CHÍNH 2]. Chúng tôi khuyến nghị [KHUYẾN NGH]

---

### 3. Giới Thiệu (1-2 trang)
**Phần:**
- Lịch sử về vấn đề churn khách hàng
- Bối cảnh kinh doanh và tầm quan trọng
- Tổng quan bộ dữ liệu
- Các mục tiêu nghiên cứu
- Phác thảo báo cáo

**Câu Hỏi Cần Trả Lời:**
- Tại sao dự đoán churn lại quan trọng?
- Bộ dữ liệu MLC là gì?
- Chúng ta đang điều tra cái gì?

---

### 4. Phương Pháp (2-3 trang)

#### 4.1 Mô Tả Bộ Dữ Liệu
- Bộ dữ liệu: mlc_churn
- Nguồn: Gói tidymodels R
- Bản ghi: ~3,333 khách hàng
- Đặc trưng: ~20 biến
- Mục tiêu: churn (yes/no)
- Phân phối lớp: tỷ lệ churn ~15%

#### 4.2 Lựa Chọn Đặc Trưng
- Kết quả phân tích tương quan
- Các đặc trưng bị loại bỏ do đa cộng tuyến
- Bộ đặc trưng cuối cùng (danh sách và lý do)
- Các bước xử lý trước (mã hóa, mở rộng quy mô, v.v.)

#### 4.3 Mô Hình Random Forest Cơ Sở
- Mô tả thuật toán
- Các siêu tham số được sử dụng
- Chỉ số đánh giá (điểm F1, tại sao lớp dương = "yes")
- Hiệu suất mô hình cơ sở

#### 4.4 Thiết Kế Thực Nghiệm

**Thực Nghiệm 1: CRD (Thiết Kế Ngẫu Nhiên Hoàn Toàn)**
- Yếu tố: k (số lượng fold)
- Mức: k = 3, 5, 10
- Lần lặp: 10
- Nhân bản: Xác thực chéo k-fold phân tầng
- Tổng thực nghiệm: 30
- Kiểm tra thống kê: Kiểm tra Levene, ANOVA một chiều, Tukey HSD

**Thực Nghiệm 2: CRFD (Thiết Kế Giai Thừa Ngẫu Nhiên Hoàn Toàn)**
- Yếu tố A: k (3, 5, 10)
- Yếu tố B: max_depth (3, 5, None)
- Lần lặp: 10
- Nhân bản: Xác thực chéo k-fold phân tầng
- Tổng thực nghiệm: 90
- Kiểm tra thống kê: ANOVA hai chiều, phân tích tương tác

#### 4.5 Phương Pháp Phân Tích Thống Kê
- Kiểm tra Levene cho đồng nhất phương sai
- ANOVA cho yếu tố duy nhất
- ANOVA hai chiều cho thiết kế giai thừa
- Tukey HSD cho so sánh từng cặp
- Kiểm tra giả định và các giải pháp thay thế

---

### 5. Kết Quả (4-6 trang)

#### 5.1 Kết Quả Mô Hình Cơ Sở
Bảng và biểu đồ hiển thị:
- Hiệu suất mô hình (F1, độ chính xác, khôi phục)
- Tầm quan trọng đặc trưng (10 hàng đầu)
- Chỉ số xác thực chéo

#### 5.2 Kết Quả Thực Nghiệm CRD

**5.2.1 Kiểm Tra Levene**
- Phương sai cho mỗi giá trị k
- Thống kê kiểm tra và giá trị p
- Kết luận về đồng nhất phương sai

**5.2.2 Thống Kê Mô Tả**
Bảng định dạng:
| k | n | Trung Bình F1 | SD | 95% CI |
|---|---|----------|----|----|

**5.2.3 ANOVA Một Chiều**
- Bảng ANOVA (SS, df, MS, F, p-value)
- Kết luận chính về ảnh hưởng k

**5.2.4 So Sánh Tukey HSD**
- Sự khác biệt từng cặp và giá trị p
- Cặp nào khác biệt đáng kể

**5.2.5 Trực Quan Hóa**
- Biểu đồ hộp hoặc violin hiển thị phân phối F1
- Biểu đồ thanh với trung bình và khoảng tin cậy 95%
- Chứng minh sự khác biệt giữa các nhóm

**5.2.6 Tóm Tắt & Giải Thích**
- k có ảnh hưởng đáng kể đến hiệu suất F1 không?
- Giá trị k nào hoạt động tốt nhất?
- Tác động thực tế

#### 5.3 Kết Quả Thực Nghiệm CRFD

**5.3.1 Thống Kê Mô Tả**
Bảng các trung bình theo kết hợp (k, max_depth):

| k | max_depth | n | Trung Bình F1 | SD | 95% CI |
|---|-----------|---|----------|----|----|
| 3 | 3 | 30 | | | |
| 3 | 5 | 30 | | | |
| ... | ... | ... | ... | ... | ... |

**5.3.2 ANOVA Hai Chiều**
- Bảng ANOVA (tất cả hiệu ứng)
- Ảnh hưởng chính của k: ý nghĩa và kết luận
- Ảnh hưởng chính của max_depth: ý nghĩa và kết luận
- Ảnh hưởng tương tác (k × max_depth): ý nghĩa và kết luận
- Kích thước hiệu ứng (partial η²)

**5.3.3 Phân Tích Tương Tác**
- Giải thích tương tác (nếu có ý nghĩa)
- Bản chất của tương tác (mô tả cách các hiệu ứng kết hợp)
- Hiệu ứng chính đơn giản (nếu cần)

**5.3.4 Trực Quan Hóa**
- Biểu đồ hiệu ứng chính (k và max_depth riêng)
- Biểu đồ tương tác (các đường cho một yếu tố, trục x yếu tố khác)
- Bản đồ nhiệt hiển thị trung bình cho tất cả kết hợp
- Bao gồm dải khoảng tin cậy

**5.3.5 Tóm Tắt & Giải Thích**
- Ảnh hưởng của cả hai yếu tố
- Ý nghĩa tương tác
- Cấu hình tốt nhất (k, max_depth)
- Hướng dẫn thực tế cho các nhà thực hành

#### 5.4 So Sánh Các Thực Nghiệm
- Kết quả CRD so với CRFD
- Xem xét cả hai yếu tố thay đổi các kết luận như thế nào?
- Sự cân bằng giữa các yếu tố

---

### 6. Thảo Luận (1-2 trang)

#### 6.1 Tóm Tắt Các Phát Hiện Chính
- Ảnh hưởng chính của k
- Ảnh hưởng chính của max_depth
- Ảnh hưởng tương tác (nếu có)

#### 6.2 Giải Thích
- Tại sao kết quả hiển thị mô hình này?
- Kết quả liên quan đến lý thuyết ML như thế nào?
- Ý nghĩa thực tế là gì?

#### 6.3 So Sánh với Văn Học
- Kết quả so sánh với các nghiên cứu ML điển hình như thế nào?
- Các phát hiện có dự kiến hay bất ngờ không?

#### 6.4 Hạn Chế
- Hạn chế bộ dữ liệu
- Hạn chế thiết kế thực nghiệm
- Giả định và các vi phạm tiềm ẩn
- Mối quan tâm về khả năng tổng quát hóa

#### 6.5 Khuyến Nghị
- Cấu hình tốt nhất cho vấn đề này
- Khi nào sử dụng k=3 vs k=5 vs k=10?
- Làm cách nào để chọn max_depth?
- Hướng dẫn thực tế cho các nhà thực hành

---

### 7. Kết Luận (½ trang)
- Tóm tắt các phát hiện chính
- Kết luận chính về các câu hỏi thực nghiệm
- Ý nghĩa cho dự đoán churn
- Hướng nghiên cứu tương lai

---

### 8. Tài Liệu Tham Khảo
- Bộ dữ liệu và gói được sử dụng
- Trích dẫn phương pháp thống kê
- Tài liệu tham khảo thuật toán ML

**Ví dụ định dạng:**
[1] M. Kuhn, "modeldata: data sets useful for modeling examples," 2025. R package version 1.5.1.
[2] L. Breiman, "Random forests," Machine Learning, vol. 45, pp. 5–32, 2001.
[3] J. Lawson, Design and analysis of experiments with R. CRC Press, 2015.

---

### 9. Phụ Lục (Tùy Chọn)
- Bảng ANOVA hoàn chỉnh
- Xác minh giả định kiểm tra thống kê
- Biểu đồ chẩn đoán
- Đoạn mã
- Trực quan hóa bổ sung

---

## Quy Trình Tạo Báo Cáo

### 1. Cách Tiếp Cận Cơ Bản (`04_report_generator_baseline.py`)
- Tải tất cả kết quả (file CRD và CRFD CSV)
- Tạo các bảng thống kê cơ bản
- Tạo các biểu đồ đơn giản
- Xuất sang markdown hoặc LaTeX

### 2. Cách Tiếp Cận Cải Tiến (`04_report_generator_improved.py`)
- Tạo hình ảnh chuyên nghiệp
- Định dạng bảng thống kê
- Tóm tắt thống kê
- Xuất hình ảnh chất lượng cao

### 3. Mẫu Báo Cáo (`report_template.md`)
- Mẫu Markdown với tất cả các phần
- Giữ chỗ cho kết quả
- Sẵn sàng điền vào các phát hiện

### 4. Tạo PDF
Tùy chọn:
- **Tùy Chọn A:** Sử dụng pandoc để chuyển đổi markdown sang PDF
  ```bash
  pandoc report.md -o report.pdf --pdf-engine=pdflatex
  ```
- **Tùy Chọn B:** Sử dụng gói Python (reportlab, pypdf, v.v.)
- **Tùy Chọn C:** Sử dụng Jupyter notebook và xuất sang PDF
- **Tùy Chọn D:** Viết LaTeX trực tiếp

---

## Biểu Đồ và Bảng Cần Thiết

### Biểu Đồ Bắt Buộc
1. ✓ Biểu đồ tầm quan trọng đặc trưng mô hình cơ sở
2. ✓ Biểu đồ hộp CRD (k=3 vs 5 vs 10)
3. ✓ Biểu đồ CRD với 95% CI
4. ✓ Biểu đồ hiệu ứng chính CRFD cho k
5. ✓ Biểu đồ hiệu ứng chính CRFD cho max_depth
6. ✓ Biểu đồ tương tác CRFD (các đường cho mỗi k)
7. ✓ Bản đồ nhiệt CRFD của các trung bình

### Bảng Bắt Buộc
1. ✓ Danh sách đặc trưng và lý do lựa chọn
2. ✓ Hiệu suất mô hình cơ sở
3. ✓ Thống kê mô tả CRD
4. ✓ Kết quả ANOVA CRD
5. ✓ So sánh Tukey HSD CRD
6. ✓ Trung bình CRFD theo kết hợp yếu tố
7. ✓ Kết quả ANOVA CRFD
8. ✓ Tóm tắt các kết luận

---

## Mẹo Viết

### Sự Rõ Ràng
- Các câu rõ ràng, ngắn gọn
- Giải thích các khái niệm thống kê cho đối tượng chung
- Sử dụng giọng chủ động
- Tránh thuật ngữ chuyên biệt hoặc định nghĩa nó

### Tổ Chức
- Luồng logic từ vấn đề → phương pháp → kết quả → kết luận
- Các tiêu đề phần rõ ràng
- Ký hiệu và thuật ngữ nhất quán

### Biểu Đồ và Bảng
- Chú thích hình/bảng giải thích nội dung
- Tham chiếu trong văn bản: "Như thể hiện trong Hình 1..."
- Bao gồm các chú thích và nhãn trục
- Sử dụng các sơ đồ màu nhất quán

### Báo Cáo Kết Quả
Luôn bao gồm:
- **Ước tính điểm** (trung bình, trung vị)
- **Biến thiên** (độ lệch chuẩn, SD, CI)
- **Kích thước mẫu** (n)
- **Kiểm tra thống kê** (nếu áp dụng)
- **Giá trị P hoặc CI**

Ví dụ:
> Điểm F1 trung bình cho k=10 là 0.620 (SD = 0.042, 95% CI: [0.604-0.636], n=30), cao hơn đáng kể so với k=3 (trung bình = 0.555, SD = 0.045, p < 0.05, Tukey HSD).

---

## Danh Sách Kiểm Tra Nộp

✓ Trang tiêu đề có tất cả thông tin bắt buộc  
✓ Tóm tắt (rõ ràng và ngắn gọn)  
✓ Phần phương pháp hoàn chỉnh  
✓ Tất cả kết quả từ cả hai thực nghiệm  
✓ Kết quả kiểm tra thống kê và giải thích  
✓ Biểu đồ và bảng chất lượng cao  
✓ Thảo luận về các phát hiện  
✓ Khuyến nghị thực tế  
✓ Kết luận  
✓ Tài liệu tham khảo  
✓ Định dạng PDF  
✓ Kết quả có thể tái tạo (seed=1234 được ghi lại)  
✓ Trình bày chuyên nghiệp  

---

## Ví Dụ Tóm Tắt Kết Quả Báo Cáo

### Thống Kê CRD để Báo Cáo:
```
ANOVA một chiều về điểm F1 theo giá trị k cho thấy ảnh hưởng đáng kể của k đến hiệu suất mô hình, F(2,27) = 5.43, p = 0.012, η² = 0.28.

Kiểm tra Tukey HSD hậu hoc cho thấy k=10 (M=0.620, SD=0.042) tạo ra điểm F1 cao hơn đáng kể so với k=3 (M=0.555, SD=0.045, p=0.008) và k=5 (M=0.587, SD=0.041, p=0.031). Không tìm thấy sự khác biệt đáng kể giữa k=3 và k=5 (p=0.187).
```

### Thống Kê CRFD để Báo Cáo:
```
ANOVA hai chiều cho thấy ảnh hưởng chính đáng kể của cả k, F(2,81)=4.82, p=0.010, η²=0.11, và max_depth, F(2,81)=8.51, p<0.001, η²=0.17. Tuy nhiên, tương tác giữa k và max_depth không có ý nghĩa thống kê, F(4,81)=1.23, p=0.302.

Ảnh hưởng của max_depth tương đối nhất quán trên tất cả các giá trị k, với max_depth=None thường tạo ra điểm F1 cao nhất (trung bình=0.618, SD=0.044) so với max_depth=3 (trung bình=0.562, SD=0.043) và max_depth=5 (trung bình=0.593, SD=0.041).
```

---

## File để Chuẩn Bị

```
Phase_4_Report/
├── README.md (file này)
├── 04_report_generator_baseline.py    # Tạo các thành phần báo cáo cơ bản
├── 04_report_generator_improved.py    # Tạo báo cáo nâng cao
├── report_template.md                 # Mẫu báo cáo markdown
├── generated_figures/
│   ├── baseline_feature_importance.png
│   ├── crd_boxplot.png
│   ├── crd_means_ci.png
│   ├── crfd_main_effects_k.png
│   ├── crfd_main_effects_maxdepth.png
│   ├── crfd_interaction_plot.png
│   ├── crfd_heatmap.png
│   └── [biểu đồ khác]
└── MLC_Churn_Report.pdf              # Báo cáo cuối cùng (sẽ tạo)
```

---

## Hướng Dẫn Chạy

```bash
# Tạo các thành phần báo cáo
python 04_report_generator_baseline.py

# Tạo nâng cao
python 04_report_generator_improved.py

# Tạo PDF (sử dụng pandoc)
pandoc report_template.md -o MLC_Churn_Report.pdf --pdf-engine=pdflatex

# Hoặc sử dụng Python:
# python -c "import pypdf; ..."  # (nếu sử dụng Python cho tạo PDF)
```

---

## Yêu Cầu Nộp

**Định Dạng File:**
- Tài liệu PDF
- Tên file: `MLC_Churn_Report.pdf`

**File Hỗ Trợ:**
- `crd_results.csv` (từ Phase 2)
- `crfd_results.csv` (từ Phase 3)

**Nội Dung:**
- Phân tích toàn diện về tất cả các thực nghiệm
- Kết quả thống kê với mức ý nghĩa
- Trực quan hóa chuyên nghiệp
- Kết luận rõ ràng và khuyến nghị

---

## Ghi Chú Cuối Cùng

1. **Dựa Trên Dữ Liệu:** Tất cả các yêu cầu được hỗ trợ bởi kết quả và thống kê
2. **Có Thể Tái Tạo:** Bao gồm seed=1234 và chi tiết xử lý trước
3. **Chuyên Nghiệp:** Viết rõ ràng và trình bày
4. **Hoàn Chỉnh:** Trả lời tất cả các câu hỏi thực nghiệm
5. **Sâu Sắc:** Cung cấp hướng dẫn thực tế cho các nhà thực hành

---

**Thời Gian Hoàn Thành Dự Kiến:** 2-3 giờ  
**Mức Độ Khó:** Vừa phải (tổng hợp công việc trước đó)

**Points:** 1 point

---

## Report Overview

### Purpose
Document the complete machine learning study on customer churn prediction, including:
- Model development process
- Experimental design and methodology
- Analysis results
- Statistical findings
- Conclusions and recommendations

### Format
- PDF document
- Length: 10-15 pages (recommended)
- Professional structure with sections, figures, and tables
- Clear writing in English or Vietnamese (specify)

### Audience
- Academic: Instructors and peers
- Technical: Readers familiar with ML and statistics
- Decision-makers: Want practical insights

---

## Report Structure

### 1. Title Page
- Title: "Customer Churn Prediction: Design and Analysis of Experiments"
- Author name and student ID
- Course name
- Date
- Institution

---

### 2. Abstract (½ page)
**Content:**
- Concise summary of the problem
- Methodology overview
- Key findings
- Main conclusion

**Example:**
> This study investigates customer churn prediction using Random Forest classification on the MLC telecom dataset. We designed two experiments using completely randomized designs (CRD and CRFD) to analyze how k-fold cross-validation and Random Forest max_depth hyperparameter affect model performance (F1 score). Results show that [KEY FINDING 1] and [KEY FINDING 2]. We recommend [RECOMMENDATION].

---

### 3. Introduction (1-2 pages)
**Sections:**
- Background on customer churn problem
- Business context and importance
- Dataset overview
- Research objectives
- Report outline

**Questions to Answer:**
- Why is churn prediction important?
- What is the MLC dataset?
- What are we investigating?

---

### 4. Methodology (2-3 pages)

#### 4.1 Dataset Description
- Dataset: mlc_churn
- Source: tidymodels R package
- Records: ~3,333 customers
- Features: ~20 variables
- Target: churn (yes/no)
- Class distribution: churn rate ~15%

#### 4.2 Feature Selection
- Correlation analysis results
- Features removed due to multicollinearity
- Final feature set (list and rationale)
- Preprocessing steps (encoding, scaling, etc.)

#### 4.3 Baseline Random Forest Model
- Algorithm description
- Hyperparameters used
- Evaluation metric (F1 score, why positive class = "yes")
- Baseline model performance

#### 4.4 Experimental Designs

**Experiment 1: CRD (Completely Randomized Design)**
- Factor: k (number of folds)
- Levels: k = 3, 5, 10
- Repeats: 10
- Replication: Stratified k-fold cross-validation
- Total experiments: 30
- Statistical tests: Levene's test, One-way ANOVA, Tukey HSD

**Experiment 2: CRFD (Completely Randomized Factorial Design)**
- Factor A: k (3, 5, 10)
- Factor B: max_depth (3, 5, None)
- Repeats: 10
- Replication: Stratified k-fold cross-validation
- Total experiments: 90
- Statistical tests: Two-way ANOVA, interaction analysis

#### 4.5 Statistical Analysis Methods
- Levene's test for variance homogeneity
- One-way ANOVA for single factor
- Two-way ANOVA for factorial design
- Tukey HSD for pairwise comparisons
- Assumptions and validations

---

### 5. Results (4-6 pages)

#### 5.1 Baseline Model Results
Table and plot showing:
- Model performance (F1, precision, recall)
- Feature importance (top 10 features)
- Cross-validation metrics

#### 5.2 CRD Experiment Results

**5.2.1 Levene's Test**
- Variance for each k value
- Test statistic and p-value
- Conclusion on variance homogeneity

**5.2.2 Descriptive Statistics**
Table format:
| k | n | Mean F1 | SD | 95% CI |
|---|---|---------|----|----|

**5.2.3 One-way ANOVA**
- ANOVA table (SS, df, MS, F, p-value)
- Main conclusion on k effect

**5.2.4 Tukey HSD Comparisons**
- Pairwise differences and p-values
- Which pairs significantly differ

**5.2.5 Visualization**
- Box plot or violin plot showing F1 distributions
- Bar plot with means and 95% confidence intervals
- Demonstrate group differences visually

**5.2.6 Summary and Interpretation**
- Does k significantly affect F1 performance?
- Which k value performs best?
- Practical implications

#### 5.3 CRFD Experiment Results

**5.3.1 Descriptive Statistics**
Table of means by (k, max_depth) combination:

| k | max_depth | n | Mean F1 | SD | 95% CI |
|---|-----------|---|---------|----|----|
| 3 | 3 | 30 | | | |
| 3 | 5 | 30 | | | |
| ... | ... | ... | ... | ... | ... |

**5.3.2 Two-Way ANOVA**
- ANOVA table (all effects)
- Main effect of k: significance and conclusion
- Main effect of max_depth: significance and conclusion
- Interaction effect (k × max_depth): significance and conclusion
- Effect sizes (partial η²)

**5.3.3 Interaction Analysis**
- Interpretation of interaction (if significant)
- Nature of interaction (describe how effects combine)
- Simple main effects (if needed)

**5.3.4 Visualizations**
- Main effects plots (k and max_depth separately)
- Interaction plots (lines for one factor, x-axis other factor)
- Heatmap showing means for all combinations
- Include confidence intervals

**5.3.5 Summary and Interpretation**
- Effects of both factors
- Significance of interaction
- Best configuration (k, max_depth)
- Practical recommendations

#### 5.4 Comparison of Experiments
- CRD findings vs CRFD findings
- How does considering both factors change conclusions?
- Trade-offs between factors

---

### 6. Discussion (1-2 pages)

#### 6.1 Key Findings Summary
- Main effect of k
- Main effect of max_depth
- Interaction effects (if present)

#### 6.2 Interpretation
- Why do results show this pattern?
- How do findings relate to ML theory?
- What is the practical significance?

#### 6.3 Comparison with Literature
- How do results compare to typical ML studies?
- Are findings expected or surprising?

#### 6.4 Limitations
- Dataset limitations
- Experimental design limitations
- Assumptions and potential violations
- Generalizability concerns

#### 6.5 Recommendations
- Best configuration for this problem
- When to use k=3 vs k=5 vs k=10?
- How should max_depth be chosen?
- Practical guidance for practitioners

---

### 7. Conclusion (½ page)
- Summary of key findings
- Main conclusion about experimental questions
- Implications for churn prediction
- Future work directions

---

### 8. References
- Datasets and packages used
- Statistical methods citations
- ML algorithm references

**Example format:**
[1] M. Kuhn, "modeldata: data sets useful for modeling examples," 2025. R package version 1.5.1.
[2] L. Breiman, "Random forests," Machine Learning, vol. 45, pp. 5–32, 2001.
[3] J. Lawson, Design and analysis of experiments with R. CRC Press, 2015.

---

### 9. Appendices (Optional)
- Complete ANOVA tables
- Statistical test assumptions verification
- Diagnostic plots
- Code snippets
- Additional visualizations

---

## Report Generation Workflow

### 1. Baseline Approach (`04_report_generator_baseline.py`)
- Load all results (CRD and CRFD CSV files)
- Generate basic statistics tables
- Create simple plots
- Export to markdown or LaTeX

### 2. Improved Approach (`04_report_generator_improved.py`)
- Professional figure generation
- Statistical table formatting
- Summary statistics
- Export high-quality images

### 3. Report Template (`report_template.md`)
- Markdown template with all sections
- Placeholder for results
- Ready to fill in with findings

### 4. PDF Generation
Options:
- **Option A:** Use pandoc to convert markdown to PDF
  ```bash
  pandoc report.md -o report.pdf --pdf-engine=pdflatex
  ```
- **Option B:** Use Python package (reportlab, pypdf, etc.)
- **Option C:** Use Jupyter notebook and export to PDF
- **Option D:** Write LaTeX directly

---

## Figures and Tables Needed

### Required Figures
1. ✓ Baseline model feature importance plot
2. ✓ CRD box plot (k=3 vs 5 vs 10)
3. ✓ CRD means with 95% CI
4. ✓ CRFD main effects plot for k
5. ✓ CRFD main effects plot for max_depth
6. ✓ CRFD interaction plot (lines for each k)
7. ✓ CRFD heatmap of means

### Required Tables
1. ✓ Feature list and selection rationale
2. ✓ Baseline model performance
3. ✓ CRD descriptive statistics
4. ✓ CRD ANOVA results
5. ✓ CRD Tukey HSD comparisons
6. ✓ CRFD means by factor combinations
7. ✓ CRFD ANOVA results
8. ✓ Summary of conclusions

---

## Writing Tips

### Clarity
- Clear, concise sentences
- Explain statistical concepts for general audience
- Use active voice
- Avoid jargon or define it

### Organization
- Logical flow from problem → methods → results → conclusion
- Clear section headings
- Consistent notation and terminology

### Figures and Tables
- Figure/table captions explain content
- Reference in text: "As shown in Figure 1..."
- Include legends and axis labels
- Use consistent color schemes

### Results Reporting
Always include:
- **Point estimate** (mean, median)
- **Variability** (std dev, SD, CI)
- **Sample size** (n)
- **Statistical test** (if applicable)
- **P-value or CI**

Example:
> The mean F1 score for k=10 was 0.620 (SD = 0.042, 95% CI: [0.604-0.636], n=30), significantly higher than k=3 (mean = 0.555, SD = 0.045, p < 0.05, Tukey HSD).

---

## Submission Checklist

✓ Title page with all required information  
✓ Abstract (clear and concise)  
✓ Complete methodology section  
✓ All results from both experiments  
✓ Statistical test results and interpretations  
✓ High-quality figures and tables  
✓ Discussion of findings  
✓ Practical recommendations  
✓ Conclusion  
✓ References  
✓ PDF format  
✓ Reproducible results (seed=1234 documented)  
✓ Professional presentation  

---

## Example Report Statistics Summary

### What to Report from Results

**CRD Phase:**
```
One-way ANOVA on F1 scores by k value revealed a significant effect of k 
on model performance, F(2,27) = 5.43, p = 0.012, η² = 0.28. 

Post-hoc Tukey HSD test indicated that k=10 (M=0.620, SD=0.042) produced 
significantly higher F1 scores than k=3 (M=0.555, SD=0.045, p=0.008) and 
k=5 (M=0.587, SD=0.041, p=0.031). No significant difference was found 
between k=3 and k=5 (p=0.187).
```

**CRFD Phase:**
```
Two-way ANOVA revealed significant main effects of both k, F(2,81)=4.82, 
p=0.010, η²=0.11, and max_depth, F(2,81)=8.51, p<0.001, η²=0.17. 
However, the interaction between k and max_depth was not statistically 
significant, F(4,81)=1.23, p=0.302.

The effect of max_depth was relatively consistent across all k values, 
with max_depth=None generally producing the highest F1 scores (mean=0.618, 
SD=0.044) compared to max_depth=3 (mean=0.562, SD=0.043) and max_depth=5 
(mean=0.593, SD=0.041).
```

---

## Files to Prepare

```
Phase_4_Report/
├── README.md (this file)
├── 04_report_generator_baseline.py    # Generate basic report components
├── 04_report_generator_improved.py    # Enhanced report generation
├── report_template.md                 # Report markdown template
├── generated_figures/
│   ├── baseline_feature_importance.png
│   ├── crd_boxplot.png
│   ├── crd_means_ci.png
│   ├── crfd_main_effects_k.png
│   ├── crfd_main_effects_maxdepth.png
│   ├── crfd_interaction_plot.png
│   ├── crfd_heatmap.png
│   └── [other figures]
└── MLC_Churn_Report.pdf              # Final report (to be generated)
```

---

## Run Instructions

```bash
# Generate report components
python 04_report_generator_baseline.py

# Enhanced generation
python 04_report_generator_improved.py

# Create PDF (using pandoc)
pandoc report_template.md -o MLC_Churn_Report.pdf --pdf-engine=pdflatex

# Or using Python:
# python -c "import pypdf; ..."  # (if using Python for PDF generation)
```

---

## Submission Requirements

**File Format:**
- PDF document
- Filename: `MLC_Churn_Report.pdf`

**Supporting Files:**
- `crd_results.csv` (from Phase 2)
- `crfd_results.csv` (from Phase 3)

**Content:**
- Comprehensive analysis of all experiments
- Statistical findings with significance levels
- Professional visualizations
- Clear conclusions and recommendations

---

## Final Notes

1. **Data-Driven:** All claims backed by results and statistics
2. **Reproducible:** Include seed=1234 and preprocessing details
3. **Professional:** Polished writing and presentation
4. **Complete:** Answer all experimental questions
5. **Insightful:** Provide practical guidance for practitioners

---

**Expected Completion Time:** 2-3 hours  
**Difficulty Level:** Medium (synthesizing previous work)
