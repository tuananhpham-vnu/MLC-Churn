# Phase 2: Thực Nghiệm CRD (Thiết Kế Ngẫu Nhiên Hoàn Toàn)

**Mục tiêu:** Thiết kế và phân tích thực nghiệm với các giá trị k-fold khác nhau trong xác thực chéo k-fold lặp lại

**Điểm:** 3 điểm (1 + 2)

---

## Thiết Kế Thực Nghiệm

### Các Yếu Tố
- **Yếu Tố:** k (số lượng fold)
- **Mức:** k = 3, 5, 10
- **Lần Lặp Lại:** 10 lần lặp lại cho mỗi giá trị k
- **Chiến Lược Fold:** Phân tầng (để xử lý mất cân bằng lớp)
- **Random Seed:** 1234
- **Tổng Lần Chạy:** 3 k-values × 10 lần lặp = 30 lần huấn luyện mô hình

### Biến Phản Ứng
- **Chỉ Số:** Điểm F1 (lớp dương = "yes")
- **Nguồn:** Xác thực chéo 10-fold phân tầng trong mỗi thiết lập k-fold

### Ngẫu Nhiên Hóa
- Xáo trộn dữ liệu trước mỗi fold
- Sử dụng chia phân tầng (duy trì tỷ lệ churn trong mỗi fold)
- Đặt random_state=1234 để tái tạo

---

## Nhiệm Vụ 1: So Sánh Phương Sai (1 điểm)

### Mục Tiêu
Xác định xem phương sai F1 có khác nhau đáng kể giữa các giá trị k không

### Phương Pháp: Kiểm Tra Levene
```python
from scipy.stats import levene

# Giả Thuyết Rỗng (H0): Phương sai bằng nhau trên các nhóm
# Giả Thuyết Thay Thế (H1): Ít nhất một phương sai khác nhau

# Thực hiện kiểm tra Levene
statistic, p_value = levene(f1_k3, f1_k5, f1_k10)

# Quy Tắc Quyết Định:
# - Nếu p-value > 0.05: Không bác bỏ H0 (phương sai bằng nhau)
# - Nếu p-value < 0.05: Bác bỏ H0 (phương sai khác nhau)
```

### Kết Quả Cần Báo Cáo
- Thống kê kiểm tra Levene
- Giá trị p
- Kết luận: Phương sai có khác nhau đáng kể hay không?
- Phương sai cho mỗi giá trị k

### Định Dạng Đầu Ra
```
==================== KẾT QUẢ KIỂM TRA LEVENE ====================
Phương sai k=3:   0.XXXX
Phương sai k=5:   0.XXXX
Phương sai k=10:  0.XXXX

Thống Kê Kiểm Tra Levene: X.XXXX
Giá Trị P: 0.XXXX

Kết Luận: [Phương sai bằng nhau / Phương sai khác nhau đáng kể]
```

---

## Nhiệm Vụ 2: Đánh Giá Tác Động của k (2 điểm)

### Mục Tiêu
Xác định xem giá trị k có ảnh hưởng đáng kể đến hiệu suất mô hình (điểm F1)

### Phương Pháp

#### 1. ANOVA Một Chiều
```python
from scipy.stats import f_oneway

# Giả Thuyết Rỗng (H0): Điểm F1 trung bình bằng nhau trên các giá trị k
# Giả Thuyết Thay Thế (H1): Ít nhất một trung bình khác nhau

f_statistic, p_value = f_oneway(f1_k3, f1_k5, f1_k10)

# Quyết Định:
# - Nếu p-value < 0.05: Bác bỏ H0 (k có ảnh hưởng đáng kể)
# - Nếu p-value > 0.05: Không bác bỏ H0 (k không có ảnh hưởng)
```

#### 2. Kiểm Tra Hậu Hoc Tukey HSD
```python
from scipy.stats import tukey_hsd

# Thực hiện so sánh từng cặp giữa tất cả các giá trị k
# Kết quả cho thấy:
# - Sự khác biệt từng cặp
# - Khoảng tin cậy
# - Cặp nào khác biệt đáng kể
res = tukey_hsd(f1_k3, f1_k5, f1_k10)
```

### Kết Quả Cần Báo Cáo
1. **Thống Kê Mô Tả**
   - Trung bình F1 cho mỗi k (với độ lệch chuẩn)
   - Khoảng tin cậy 95%
   - Min/Max F1 cho mỗi k

2. **Kết Quả ANOVA**
   - Thống kê F
   - Giá trị P
   - Kết luận (ảnh hưởng đáng kể hay không)

3. **Kết Quả Tukey HSD**
   - So sánh từng cặp
   - Khoảng tin cậy cho các sự khác biệt
   - Cặp nào khác biệt đáng kể

### Định Dạng Đầu Ra
```
==================== KẾT QUẢ THỰC NGHIỆM CRD ====================

Thống Kê Mô Tả:
k=3  | Trung Bình: 0.XXXX | ĐL: 0.XXXX | 95% CI: [0.XXXX - 0.XXXX]
k=5  | Trung Bình: 0.XXXX | ĐL: 0.XXXX | 95% CI: [0.XXXX - 0.XXXX]
k=10 | Trung Bình: 0.XXXX | ĐL: 0.XXXX | 95% CI: [0.XXXX - 0.XXXX]

ANOVA Một Chiều:
Thống kê F: X.XXXX
Giá trị P: 0.XXXX
Kết luận: [k có ảnh hưởng đáng kể / k không có ảnh hưởng]

Kiểm Tra Hậu Hoc Tukey HSD:
3 vs 5:   Sự Khác Biệt = 0.XXXX, p-value = 0.XXXX
3 vs 10:  Sự Khác Biệt = 0.XXXX, p-value = 0.XXXX
5 vs 10:  Sự Khác Biệt = 0.XXXX, p-value = 0.XXXX

Các Cặp Khác Biệt: [Liệt kê các cặp khác nhau]
```

---

## Cách Tiếp Cận Cơ Bản (`02_crd_baseline.py`)

```python
# 1. Tải dữ liệu được xử lý trước từ Phase 1
# 2. Đặt random seed = 1234

# 3. Cho mỗi k trong [3, 5, 10]:
#    cho mỗi repeat trong phạm vi(10):
#        - Tạo StratifiedKFold(n_splits=k, random_state=seed+repeat)
#        - Huấn luyện mô hình RF trên mỗi fold
#        - Thu thập điểm số F1
#        - Lưu trữ kết quả

# 4. Sắp xếp kết quả
#    - Tạo DataFrame với các cột: [k, repeat, f1_score]
#    - Lưu vào crd_results.csv

# 5. Thực hiện kiểm tra Levene
#    - So sánh phương sai trên các giá trị k
#    - In kết quả

# 6. Thực hiện ANOVA một chiều
#    - Kiểm tra xem k có ảnh hưởng đến trung bình F1
#    - In kết quả

# 7. Thực hiện Tukey HSD
#    - So sánh từng cặp
#    - In kết quả

# 8. Tạo biểu đồ so sánh
#    - Biểu đồ hộp hiển thị phân phối F1 cho mỗi k
#    - Thêm trung bình và khoảng tin cậy
#    - Lưu biểu đồ
```

**Đầu Ra:**
- `crd_results.csv` - Kết quả thô
- Kết quả kiểm tra thống kê bàn phím lệnh
- Biểu đồ so sánh hộp

---

## Cách Tiếp Cận Cải Tiến (`02_crd_improved.py`)

Các Cải Tiến:

### 1. Xác Thực Dữ Liệu
- Xác minh rằng chia phân tầng duy trì tỷ lệ churn
- Kiểm tra rò rỉ dữ liệu
- Xác thực kích thước fold

### 2. Kiểm Tra Thống Kê Mạnh Mẽ
- Kiểm tra giả định cho ANOVA
- Kiểm tra tính chuẩn mực (Shapiro-Wilk)
- Đồng nhất phương sai (Kiểm tra Bartlett làm giải pháp thay thế)
- Giải pháp thay thế không parametric (Kruskal-Wallis) nếu vi phạm giả định ANOVA

### 3. Báo Cáo Kích Thước Hiệu Ứng
- Eta-squared (kích thước hiệu ứng cho ANOVA)
- Cohen's d cho so sánh từng cặp
- Ý nghĩa thực tế vs thống kê

### 4. Trực Quan Hóa Toàn Diện
- Biểu đồ hộp với các điểm riêng lẻ
- Biểu đồ violin
- Biểu đồ thanh với các thanh lỗi (trung bình ± 95% CI)
- Biểu đồ Q-Q để kiểm tra tính chuẩn mực

### 5. Tài Liệu Chi Tiết
- Ghi lại tất cả các bước xử lý trước
- Lưu bảng kết quả chi tiết
- Bao gồm khoảng tin cậy trong đầu ra CSV

---

## Script Phân Tích (`02_crd_analysis.py`)

Script chuyên dụng cho phân tích thống kê:
- Tải kết quả từ `crd_results.csv`
- Thực hiện tất cả các kiểm tra thống kê
- Tạo tất cả các trực quan hóa
- Tạo báo cáo phân tích chi tiết

---

## Chiến Lược Thu Thập Dữ Liệu

### Mã Giả cho Thu Thập Dữ Liệu
```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

results = []
random_seed = 1234

for k in [3, 5, 10]:
    for repeat in range(10):
        # Tạo splitter k-fold phân tầng
        skf = StratifiedKFold(n_splits=k, 
                              shuffle=True, 
                              random_state=random_seed + repeat)
        
        # Khởi tạo mô hình
        model = RandomForestClassifier(random_state=random_seed)
        
        # Xác thực chéo với điểm F1
        f1_scores = cross_val_score(model, X, y, 
                                    cv=skf, 
                                    scoring='f1',
                                    n_jobs=-1)
        
        # Lưu kết quả
        for fold_idx, f1 in enumerate(f1_scores):
            results.append({
                'k': k,
                'repeat': repeat,
                'fold': fold_idx,
                'f1_score': f1
            })

df_results = pd.DataFrame(results)
df_results.to_csv('crd_results.csv', index=False)
```

---

## File Đầu Ra

```
Phase_2_CRD/
├── README.md (file này)
├── 02_crd_baseline.py
├── 02_crd_improved.py
├── 02_crd_analysis.py
├── crd_results.csv                      # Kết quả thô (bắt buộc nộp)
└── outputs/
    ├── levene_test_results.txt
    ├── anova_results.txt
    ├── tukey_hsd_results.txt
    ├── crd_boxplot.png
    ├── crd_violin_plot.png
    ├── crd_means_with_ci.png
    ├── crd_comprehensive_analysis.txt
    └── crd_detailed_report.pdf
```

---

## Kết Quả Dự Kiến

### Phát Hiện Điển Hình
- **Kiểm Tra Levene:** Có khả năng p > 0.05 (phương sai tương tự)
- **ANOVA:** Có khả năng p < 0.05 (k ảnh hưởng đến hiệu suất)
- **Tukey HSD:** k=10 thường tốt hơn k=3 và k=5
- **Phạm Vi F1 Điển Hình:** 0.50-0.70 tùy thuộc vào dữ liệu

### Mô Hình Hiệu Suất
- k=3: Ít fold → phương sai cao hơn (ít ổn định)
- k=5: Vừa phải
- k=10: Nhiều fold → phương sai thấp hơn (ổn định hơn)

---

## Yêu Cầu Nộp

✓ `crd_results.csv` với tất cả kết quả thực nghiệm  
✓ Thống kê kiểm tra Levene và giá trị p  
✓ Kết quả ANOVA một chiều  
✓ So sánh hậu hoc Tukey HSD  
✓ Biểu đồ so sánh hiển thị 3 nhóm  
✓ Giá trị trung bình và khoảng tin cậy 95% được báo cáo  
✓ Kết luận thống kê được ghi lại  

---

## Hướng Dẫn Chạy

```bash
# Thực nghiệm CRD cơ bản
python 02_crd_baseline.py

# Xác thực nâng cao
python 02_crd_improved.py

# Phân tích thống kê và trực quan hóa
python 02_crd_analysis.py
```

---

## Ghi Chú Quan Trọng

1. **Phân Tầng:** Luôn sử dụng StratifiedKFold để duy trì tỷ lệ churn
2. **Random Seed:** Sử dụng seed nhất quán (1234 + repeat) để tái tạo
3. **Chỉ Số:** Luôn sử dụng điểm F1 với positive_class='yes'
4. **Tài Liệu:** Báo cáo cả ý nghĩa thống kê VÀ thực tế
5. **Trực Quan Hóa:** Luôn bao gồm trung bình và khoảng tin cậy

---

## Các Bước Tiếp Theo

→ Phase 3: Thực Nghiệm CRFD (mở rộng đến hai yếu tố: k và max_depth)

**Objective:** Design and analyze experiment with varying k-fold values in repeated k-fold cross-validation

**Points:** 3 points (1 + 2)

---

## Experimental Design

### Factors
- **Factor:** k (number of folds)
- **Levels:** k = 3, 5, 10
- **Repeats:** 10 repetitions for each k value
- **Fold Strategy:** Stratified (to handle class imbalance)
- **Random Seed:** 1234
- **Total Runs:** 3 k-values × 10 repeats = 30 model trainings

### Response Variable
- **Metric:** F1 Score (positive class = "yes")
- **Source:** 10-fold stratified cross-validation within each k-fold setup

### Randomization
- Shuffle data before each fold
- Use stratified split (maintain churn ratio in each fold)
- Set random_state=1234 for reproducibility

---

## Task 1: Compare Variance (1 point)

### Objective
Determine if F1 variance differs significantly across different k values

### Method: Levene's Test
```python
from scipy.stats import levene

# Null Hypothesis (H0): Variances are equal across groups
# Alternative (H1): At least one variance is different

# Perform Levene's test
statistic, p_value = levene(f1_k3, f1_k5, f1_k10)

# Decision rule:
# - If p-value > 0.05: Fail to reject H0 (variances are equal)
# - If p-value < 0.05: Reject H0 (variances are different)
```

### Results to Report
- Levene's test statistic
- P-value
- Conclusion: Do variances differ significantly?
- Variance for each k value

### Output Format
```
==================== LEVENE'S TEST RESULTS ====================
k=3 variance:   0.XXXX
k=5 variance:   0.XXXX
k=10 variance:  0.XXXX

Levene's Test Statistic: X.XXXX
P-value: 0.XXXX

Conclusion: [Variances are equal / Variances differ significantly]
```

---

## Task 2: Evaluate k Impact (2 points)

### Objective
Determine if k value significantly affects model performance (F1 score)

### Methods

#### 1. One-way ANOVA
```python
from scipy.stats import f_oneway

# Null Hypothesis (H0): Mean F1 scores are equal across k values
# Alternative (H1): At least one mean is different

f_statistic, p_value = f_oneway(f1_k3, f1_k5, f1_k10)

# Decision:
# - If p-value < 0.05: Reject H0 (k has significant effect)
# - If p-value > 0.05: Fail to reject H0 (k has no effect)
```

#### 2. Tukey HSD Post-hoc Test
```python
from scipy.stats import tukey_hsd

# Performs pairwise comparisons between all k values
# Results show:
# - Pairwise differences
# - Confidence intervals
# - Which pairs differ significantly
res = tukey_hsd(f1_k3, f1_k5, f1_k10)
```

### Results to Report
1. **Descriptive Statistics**
   - Mean F1 for each k (with standard deviation)
   - 95% Confidence intervals
   - Min/Max F1 for each k

2. **ANOVA Results**
   - F-statistic
   - P-value
   - Conclusion (significant effect or not)

3. **Tukey HSD Results**
   - Pairwise comparisons
   - Confidence intervals for differences
   - Which pairs are significantly different

### Output Format
```
==================== CRD EXPERIMENT RESULTS ====================

Descriptive Statistics:
k=3  | Mean: 0.XXXX | SD: 0.XXXX | 95% CI: [0.XXXX - 0.XXXX]
k=5  | Mean: 0.XXXX | SD: 0.XXXX | 95% CI: [0.XXXX - 0.XXXX]
k=10 | Mean: 0.XXXX | SD: 0.XXXX | 95% CI: [0.XXXX - 0.XXXX]

One-way ANOVA:
F-statistic: X.XXXX
P-value: 0.XXXX
Conclusion: [k has significant effect / k has no effect]

Tukey HSD Post-hoc Test:
3 vs 5:   Difference = 0.XXXX, p-value = 0.XXXX
3 vs 10:  Difference = 0.XXXX, p-value = 0.XXXX
5 vs 10:  Difference = 0.XXXX, p-value = 0.XXXX

Significant pairs: [List which pairs differ]
```

---

## Baseline Approach (`02_crd_baseline.py`)

```python
# 1. Load preprocessed data from Phase 1
# 2. Set random seed = 1234

# 3. For each k in [3, 5, 10]:
#    for repeat in range(10):
#        - Create StratifiedKFold(n_splits=k, random_state=seed+repeat)
#        - Train RF model on each fold
#        - Collect F1 scores
#        - Store results

# 4. Organize results
#    - Create DataFrame with columns: [k, repeat, f1_score]
#    - Save to crd_results.csv

# 5. Perform Levene's test
#    - Compare variances across k values
#    - Print results

# 6. Perform one-way ANOVA
#    - Test if k affects F1 mean
#    - Print results

# 7. Perform Tukey HSD
#    - Pairwise comparisons
#    - Print results

# 8. Create comparison plot
#    - Box plot showing F1 distribution for each k
#    - Add means and confidence intervals
#    - Save plot
```

**Output:**
- `crd_results.csv` - Raw results
- Console statistical test results
- Box plot comparison

---

## Improved Approach (`02_crd_improved.py`)

Enhancements:

### 1. Data Validation
- Verify stratified split maintains churn ratio
- Check for data leakage
- Validate fold sizes

### 2. Robust Statistical Testing
- Assumptions checking for ANOVA
- Normality tests (Shapiro-Wilk)
- Homogeneity of variance (Bartlett's test as alternative)
- Non-parametric alternative (Kruskal-Wallis) if ANOVA assumptions violated

### 3. Effect Size Reporting
- Eta-squared (effect size for ANOVA)
- Cohen's d for pairwise comparisons
- Practical vs statistical significance

### 4. Comprehensive Visualization
- Box plots with individual points
- Violin plots
- Bar plots with error bars (mean ± 95% CI)
- Q-Q plots for normality check

### 5. Detailed Documentation
- Document all preprocessing steps
- Save detailed results table
- Include confidence intervals in CSV output

---

## Analysis Script (`02_crd_analysis.py`)

Dedicated script for statistical analysis:
- Load results from `crd_results.csv`
- Perform all statistical tests
- Generate all visualizations
- Create detailed analysis report

---

## Data Collection Strategy

### Pseudocode for Data Collection
```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

results = []
random_seed = 1234

for k in [3, 5, 10]:
    for repeat in range(10):
        # Create stratified k-fold splitter
        skf = StratifiedKFold(n_splits=k, 
                              shuffle=True, 
                              random_state=random_seed + repeat)
        
        # Initialize model
        model = RandomForestClassifier(random_state=random_seed)
        
        # Cross-validation with F1 score
        f1_scores = cross_val_score(model, X, y, 
                                    cv=skf, 
                                    scoring='f1',
                                    n_jobs=-1)
        
        # Store results
        for fold_idx, f1 in enumerate(f1_scores):
            results.append({
                'k': k,
                'repeat': repeat,
                'fold': fold_idx,
                'f1_score': f1
            })

df_results = pd.DataFrame(results)
df_results.to_csv('crd_results.csv', index=False)
```

---

## Output Files

```
Phase_2_CRD/
├── README.md (this file)
├── 02_crd_baseline.py
├── 02_crd_improved.py
├── 02_crd_analysis.py
├── crd_results.csv                      # Raw results (required for submission)
└── outputs/
    ├── levene_test_results.txt
    ├── anova_results.txt
    ├── tukey_hsd_results.txt
    ├── crd_boxplot.png
    ├── crd_violin_plot.png
    ├── crd_means_with_ci.png
    ├── crd_comprehensive_analysis.txt
    └── crd_detailed_report.pdf
```

---

## Expected Results

### Typical Findings
- **Levene's Test:** Likely p > 0.05 (variances similar)
- **ANOVA:** Likely p < 0.05 (k affects performance)
- **Tukey HSD:** k=10 typically better than k=3 and k=5
- **Typical F1 Range:** 0.50-0.70 depending on data

### Performance Pattern
- k=3: Fewer folds → more variance (less stable)
- k=5: Moderate
- k=10: More folds → lower variance (more stable)

---

## Statistical Formulas

### Levene's Test
Tests equality of variances using residuals from group means

### One-way ANOVA
$$F = \frac{MS_{between}}{MS_{within}} = \frac{\sum_{i=1}^{k} n_i(m_i - m)^2 / (k-1)}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (x_{ij} - m_i)^2 / (N-k)}$$

### Tukey HSD
$$HSD = q \sqrt{\frac{MS_{within}}{n}}$$

---

## Submission Requirements

✓ `crd_results.csv` with all experimental results  
✓ Levene's test statistic and p-value  
✓ One-way ANOVA results  
✓ Tukey HSD post-hoc comparisons  
✓ Comparison plot showing 3 groups  
✓ Means and 95% confidence intervals reported  
✓ Statistical conclusions documented  

---

## Run Instructions

```bash
# Baseline CRD experiment
python 02_crd_baseline.py

# Enhanced validation
python 02_crd_improved.py

# Statistical analysis and visualization
python 02_crd_analysis.py
```

---

## Important Notes

1. **Stratification:** Always use StratifiedKFold to maintain churn ratio
2. **Random Seed:** Use consistent seed (1234 + repeat) for reproducibility
3. **Metrics:** Always use F1 score with positive_class='yes'
4. **Documentation:** Report both statistical AND practical significance
5. **Visualization:** Always include means and confidence intervals

---

## Next Steps

→ Phase 3: CRFD Experiment (extend to two factors: k and max_depth)
