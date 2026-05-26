# Phase 3: Thực Nghiệm CRFD (Thiết Kế Giai Thừa Ngẫu Nhiên Hoàn Toàn)

**Mục tiêu:** Phân tích tương tác giữa k và max_depth bằng thiết kế giai thừa 2 yếu tố

**Điểm:** 4 điểm (3 + 1)

---

## Thiết Kế Thực Nghiệm

### Các Yếu Tố
- **Yếu Tố A (k):** k = 3, 5, 10 (3 mức)
- **Yếu Tố B (max_depth):** 3, 5, Không giới hạn/None (3 mức)
- **Lần Lặp Lại:** 10 lần lặp lại cho mỗi kết hợp
- **Chiến Lược Fold:** Phân tầng
- **Random Seed:** 1234
- **Tổng Lần Chạy:** 3 × 3 × 10 = 90 thực nghiệm

### Thiết Kế: Giai Thừa 3 × 3 với Nhân Bản
```
        max_depth=3   max_depth=5   max_depth=None
k=3        10             10            10        (30 lần chạy)
k=5        10             10            10        (30 lần chạy)
k=10       10             10            10        (30 lần chạy)
        ────────────────────────────────────────
Tổng:      30             30            30        (90 tổng cộng)
```

### Biến Phản Ứng
- **Chỉ Số:** Điểm F1 (lớp dương = "yes")

---

## Nhiệm Vụ 1: Đánh Giá Hiệu Ứng Chính (3 điểm)

### Mục Tiêu
Xác định xem k và max_depth có ảnh hưởng đáng kể đến hiệu suất mô hình

### Phân Tích Cần Thiết

#### 1. Thống Kê Mô Tả
Báo cáo cho mỗi mức yếu tố:
- Trung bình điểm F1
- Độ lệch chuẩn
- Khoảng tin cậy 95%
- Min/Max điểm F1

Ví dụ đầu ra:
```
Yếu Tố A (k):
k=3:    Trung Bình = 0.XXXX, ĐL = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]
k=5:    Trung Bình = 0.XXXX, ĐL = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]
k=10:   Trung Bình = 0.XXXX, ĐL = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]

Yếu Tố B (max_depth):
max_depth=3:     Trung Bình = 0.XXXX, ĐL = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]
max_depth=5:     Trung Bình = 0.XXXX, ĐL = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]
max_depth=None:  Trung Bình = 0.XXXX, ĐL = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]
```

#### 2. ANOVA Hai Chiều
```python
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Xây dựng mô hình ANOVA
# Công thức: F1_score ~ k + max_depth + k:max_depth
model = ols('f1_score ~ C(k) + C(max_depth) + C(k):C(max_depth)', 
            data=df_results).fit()

# Thực hiện ANOVA
anova_table = anova_lm(model, typ=2)  # ANOVA Loại II
```

#### 3. Kết Quả Cần Báo Cáo

Cho mỗi yếu tố (k và max_depth):
- Tổng Bình Phương (SS)
- Bậc Tự Do (df)
- Bình Phương Trung Bình (MS)
- Thống kê F
- Giá trị P
- **Kết Luận:** Yếu tố A/B có ảnh hưởng đáng kể (p < 0.05) hay không

### Định Dạng Đầu Ra
```
==================== KẾT QUẢ ANOVA HAI CHIỀU ====================

Nguồn Biến Thiên        SS      df    MS      F       p-value    Đáng Kể
─────────────────────────────────────────────────────────────────────────
Yếu Tố A (k)          0.XXX   2     0.XXX   X.XXX   0.XXXX     [Có/Không]
Yếu Tố B (max_depth)  0.XXX   2     0.XXX   X.XXX   0.XXXX     [Có/Không]
Tương Tác (A×B)       0.XXX   4     0.XXX   X.XXX   0.XXXX     [Có/Không]
Lỗi                   0.XXX  81     0.XXX
─────────────────────────────────────────────────────────────────────────
Tổng Cộng             0.XXX  89

Kết Luận:
- Ảnh Hưởng Yếu Tố A (k): [ĐÁng KỲ / KHÔNG ĐÁng Kỳ]
- Ảnh Hưởng Yếu Tố B (max_depth): [ĐÁng KỲ / KHÔNG ĐÁng Kỳ]
```

#### 4. Kích Thước Hiệu Ứng
Báo cáo partial eta-squared cho mỗi yếu tố:
$$\text{Partial } \eta^2 = \frac{SS_{effect}}{SS_{effect} + SS_{error}}$$

Giải Thích:
- Hiệu ứng nhỏ: 0.01
- Hiệu ứng vừa: 0.06
- Hiệu ứng lớn: 0.14+

---

## Nhiệm Vụ 2: Phân Tích Tương Tác (1 điểm)

### Mục Tiêu
Xác định xem k và max_depth có tương tác trong ảnh hưởng của chúng đến điểm F1

### Phương Pháp

#### 1. Kiểm Tra Ý Nghĩa Tương Tác
Từ kết quả ANOVA Hai Chiều, xem xét giá trị p tương tác:
- Nếu p < 0.05: Tương tác **có ý nghĩa thống kê**
- Nếu p > 0.05: Tương tác **không có ý nghĩa thống kê**

#### 2. Biểu Đồ Tương Tác
Tạo biểu đồ đường hiển thị tương tác:
```python
# Biểu đồ 1: Các đường cho mỗi giá trị k, trục x = max_depth
# Biểu đồ 2: Các đường cho mỗi max_depth, trục x = k

# Giải thích:
# - Các đường song song → Không tương tác
# - Các đường không song song → Có tương tác
# - Các đường cắt nhau → Tương tác mạnh
```

#### 3. Hiệu Ứng Chính Đơn Giản (nếu tương tác có ý nghĩa)
Nếu tương tác tồn tại, phân tích ảnh hưởng của một yếu tố ở mỗi mức của yếu tố khác:
- Ảnh hưởng của k ở max_depth=3, 5, None
- Ảnh hưởng của max_depth ở k=3, 5, 10

#### 4. Giải Thích
Mô tả cách các yếu tố tương tác với nhau

### Kết Quả Cần Báo Cáo
1. **Ý Nghĩa Tương Tác**
   - Thống kê F và giá trị p
   - Kích thước hiệu ứng (partial η²)
   - Kết luận

2. **Bản Chất của Tương Tác (nếu có ý nghĩa)**
   - Mô tả cách các yếu tố tương tác
   - Kết hợp nào hoạt động tốt nhất/tệ nhất
   - Tác động thực tế

3. **Trực Quan Hóa**
   - Biểu đồ tương tác với trung bình và thanh lỗi
   - Hiển thị khoảng tin cậy

---

## Cách Tiếp Cận Cơ Bản (`03_crfd_baseline.py`)

```python
# 1. Tải dữ liệu được xử lý trước từ Phase 1

# 2. Tạo lưới thực nghiệm
#    factors = {
#        'k': [3, 5, 10],
#        'max_depth': [3, 5, None]
#    }

# 3. Cho mỗi kết hợp các yếu tố:
#    cho mỗi repeat trong phạm vi(10):
#        - Tạo StratifiedKFold(n_splits=k)
#        - Tạo RandomForestClassifier(max_depth=max_depth, random_state=seed)
#        - Thực hiện xác thực chéo
#        - Thu thập điểm số F1
#        - Lưu trữ kết quả

# 4. Sắp xếp kết quả vào DataFrame
#    Cột: [k, max_depth, repeat, fold, f1_score]
#    Lưu vào crfd_results.csv

# 5. Tổng hợp kết quả theo (k, max_depth)
#    Tính toán trung bình F1 cho mỗi kết hợp

# 6. Thực hiện ANOVA Hai Chiều
#    Kiểm tra ảnh hưởng của k và max_depth
#    In bảng ANOVA

# 7. Tạo biểu đồ tương tác
#    Vẽ trung bình với thanh lỗi
#    Hiển thị các đường cho mỗi giá trị k
#    Lưu biểu đồ

# 8. Báo cáo kết quả
#    - Trung bình và CI cho mỗi yếu tố
#    - Kết quả ANOVA
#    - Tóm tắt tương tác
```

**Đầu Ra:**
- `crfd_results.csv` - Kết quả thô
- Bảng ANOVA và giải thích
- Biểu đồ tương tác

---

## Cách Tiếp Cận Cải Tiến (`03_crfd_improved.py`)

Các Cải Tiến:

### 1. Xác Thực Thực Nghiệm
- Xác minh các bộ phân tầng
- Kiểm tra cân bằng trong các kết hợp
- Xác thực kích thước fold
- Đảm bảo tái tạo với seed

### 2. Kiểm Tra Giả Định ANOVA
- Tính chuẩn mực (Kiểm tra Shapiro-Wilk)
- Đồng nhất phương sai (Kiểm tra Levene)
- Độc lập (đã thỏa mãn theo thiết kế)
- Báo cáo: Dữ liệu có đáp ứng giả định không?
- Giải pháp thay thế: Kruskal-Wallis nếu vi phạm giả định

### 3. Hiệu Chỉnh So Sánh Đa Lần
- Tukey HSD cho so sánh từng cặp
- Hiệu chỉnh Bonferroni nếu cần
- Báo cáo giá trị p được điều chỉnh

### 4. Kích Thước Hiệu Ứng
- Partial eta-squared
- Omega-squared
- So sánh ý nghĩa thực tế vs thống kê

### 5. Trực Quan Hóa Toàn Diện
- Biểu đồ tương tác (cả hai góc nhìn)
- Biểu đồ hiệu ứng chính
- Bản đồ nhiệt các giá trị trung bình F1
- Dải khoảng tin cậy

### 6. Chẩn Đoán Mô Hình
- Biểu đồ phần dư
- Biểu đồ Q-Q
- Biểu đồ đồng nhất
- Kiểm tra ngoại lệ

---

## Script Phân Tích (`03_crfd_analysis.py`)

Script phân tích chuyên dụng:
- Tải `crfd_results.csv`
- Thực hiện ANOVA toàn diện
- Tạo tất cả trực quan hóa
- Tạo báo cáo chi tiết
- So sánh với kết quả CRD

---

## Chiến Lược Thu Thập Dữ Liệu

### Mã Giả
```python
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

results = []
random_seed = 1234

# Tất cả các kết hợp yếu tố
k_values = [3, 5, 10]
max_depth_values = [3, 5, None]

for k, max_depth in product(k_values, max_depth_values):
    for repeat in range(10):
        # Tạo mô hình
        model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=100,
            random_state=random_seed
        )
        
        # Tạo k-fold phân tầng
        skf = StratifiedKFold(n_splits=k, 
                              shuffle=True, 
                              random_state=random_seed + repeat)
        
        # Xác thực chéo
        f1_scores = cross_val_score(model, X, y, 
                                    cv=skf, 
                                    scoring='f1',
                                    n_jobs=-1)
        
        # Lưu kết quả
        for fold_idx, f1 in enumerate(f1_scores):
            results.append({
                'k': k,
                'max_depth': max_depth,
                'repeat': repeat,
                'fold': fold_idx,
                'f1_score': f1
            })

df_results = pd.DataFrame(results)
df_results.to_csv('crfd_results.csv', index=False)
```

---

## Bảng Kết Quả Dự Kiến

Ví dụ về kết quả giai thừa:

| k   | max_depth | n   | Trung Bình F1 | Std   | 95% CI        |
|-----|-----------|-----|-----------|-------|---------------|
| 3   | 3         | 30  | 0.550     | 0.045 | [0.534-0.566] |
| 3   | 5         | 30  | 0.580     | 0.042 | [0.564-0.596] |
| 3   | None      | 30  | 0.600     | 0.048 | [0.583-0.617] |
| 5   | 3         | 30  | 0.562     | 0.043 | [0.546-0.578] |
| 5   | 5         | 30  | 0.592     | 0.041 | [0.576-0.608] |
| 5   | None      | 30  | 0.612     | 0.046 | [0.596-0.628] |
| 10  | 3         | 30  | 0.575     | 0.041 | [0.559-0.591] |
| 10  | 5         | 30  | 0.605     | 0.040 | [0.589-0.621] |
| 10  | None      | 30  | 0.625     | 0.044 | [0.609-0.641] |

---

## File Đầu Ra

```
Phase_3_CRFD/
├── README.md (file này)
├── 03_crfd_baseline.py
├── 03_crfd_improved.py
├── 03_crfd_analysis.py
├── crfd_results.csv                      # Kết quả thô (bắt buộc nộp)
└── outputs/
    ├── anova_table.txt
    ├── interaction_significance.txt
    ├── crfd_means_table.csv
    ├── main_effects_plot.png
    ├── interaction_plot_1.png             # Các đường cho mỗi k
    ├── interaction_plot_2.png             # Các đường cho mỗi max_depth
    ├── heatmap_means.png
    ├── residual_diagnostic_plots.png
    ├── crfd_comprehensive_report.txt
    └── crfd_detailed_analysis.pdf
```

---

## Yêu Cầu Nộp

✓ `crfd_results.csv` với tất cả 90 kết quả thực nghiệm  
✓ Bảng ANOVA Hai Chiều (k, max_depth, ảnh hưởng tương tác)  
✓ Phân tích hiệu ứng chính với trung bình và 95% CI  
✓ Kiểm tra ý nghĩa tương tác với giá trị p  
✓ Các biểu đồ tương tác rõ ràng hiển thị ảnh hưởng  
✓ Kích thước hiệu ứng (partial η²) cho mỗi yếu tố  
✓ Kết luận về ảnh hưởng yếu tố  
✓ Kết quả có thể tái tạo với seed=1234  

---

## Hướng Dẫn Chạy

```bash
# Thực nghiệm CRFD cơ bản
python 03_crfd_baseline.py

# Xác thực nâng cao
python 03_crfd_improved.py

# Phân tích thống kê và trực quan hóa
python 03_crfd_analysis.py
```

---

## Ghi Chú Quan Trọng

1. **Phân Tầng:** Luôn sử dụng StratifiedKFold
2. **Nhân Bản:** Cần 10 nhân bản cho mỗi kết hợp xử lý (không chỉ 1)
3. **Random Seed:** Sử dụng seed nhất quán để tái tạo
4. **Chỉ Số:** F1 score với positive_class='yes'
5. **Giả Định:** Kiểm tra và báo cáo giả định ANOVA
6. **Giải Thích:** Tập trung vào cả ý nghĩa thống kê VÀ thực tế

---

## So Sánh với Giai Đoạn CRD

| Khía Cạnh | CRD (Phase 2) | CRFD (Phase 3) |
|-----------|-----------|---------|
| Các Yếu Tố | 1 (chỉ k) | 2 (k và max_depth) |
| Mức | 3 | 3 × 3 = 9 |
| Thực Nghiệm | 30 | 90 |
| Câu Hỏi | k có ảnh hưởng đến F1 không? | k và max_depth có ảnh hưởng và tương tác không? |
| ANOVA | Một Chiều | Hai Chiều |

---

## Các Bước Tiếp Theo

→ Phase 4: Viết Báo Cáo (Tổng hợp tất cả các phát hiện)

**Objective:** Analyze interaction between k and max_depth using 2-factor factorial design

**Points:** 4 points (3 + 1)

---

## Experimental Design

### Factors
- **Factor A (k):** k = 3, 5, 10 (3 levels)
- **Factor B (max_depth):** 3, 5, None/unlimited (3 levels)
- **Repeats:** 10 repetitions per combination
- **Fold Strategy:** Stratified
- **Random Seed:** 1234
- **Total Runs:** 3 × 3 × 10 = 90 experiments

### Design Type: 3 × 3 Factorial with Replication
```
        max_depth=3   max_depth=5   max_depth=None
k=3        10             10            10        (30 runs)
k=5        10             10            10        (30 runs)
k=10       10             10            10        (30 runs)
        ────────────────────────────────────────
Total:     30             30            30        (90 total)
```

### Response Variable
- **Metric:** F1 Score (positive class = "yes")

---

## Task 1: Evaluate Main Effects (3 points)

### Objective
Determine if k and max_depth significantly affect model performance

### Analyses Required

#### 1. Descriptive Statistics
Report for each factor level:
- Mean F1 score
- Standard deviation
- 95% Confidence intervals
- Min/Max F1 scores

Example output:
```
Factor A (k):
k=3:    Mean = 0.XXXX, SD = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]
k=5:    Mean = 0.XXXX, SD = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]
k=10:   Mean = 0.XXXX, SD = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]

Factor B (max_depth):
max_depth=3:     Mean = 0.XXXX, SD = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]
max_depth=5:     Mean = 0.XXXX, SD = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]
max_depth=None:  Mean = 0.XXXX, SD = 0.XXXX, 95% CI = [0.XXXX - 0.XXXX]
```

#### 2. Two-Way ANOVA
```python
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Build ANOVA model
# Formula: F1_score ~ k + max_depth + k:max_depth
model = ols('f1_score ~ C(k) + C(max_depth) + C(k):C(max_depth)', 
            data=df_results).fit()

# Perform ANOVA
anova_table = anova_lm(model, typ=2)  # Type II ANOVA
```

#### 3. Results to Report

For each factor (k and max_depth):
- Sum of Squares (SS)
- Degrees of Freedom (df)
- Mean Square (MS)
- F-statistic
- P-value
- **Conclusion:** Factor A/B has significant effect (p < 0.05) or not

### Output Format
```
==================== TWO-WAY ANOVA RESULTS ====================

Source of Variation    SS      df    MS      F       p-value    Significant
─────────────────────────────────────────────────────────────────────────
Factor A (k)          0.XXX   2     0.XXX   X.XXX   0.XXXX     [Yes/No]
Factor B (max_depth)  0.XXX   2     0.XXX   X.XXX   0.XXXX     [Yes/No]
Interaction (A×B)     0.XXX   4     0.XXX   X.XXX   0.XXXX     [Yes/No]
Error                 0.XXX  81     0.XXX
─────────────────────────────────────────────────────────────────────────
Total                 0.XXX  89

Conclusions:
- Factor A (k) effect: [SIGNIFICANT / NOT SIGNIFICANT]
- Factor B (max_depth) effect: [SIGNIFICANT / NOT SIGNIFICANT]
```

#### 4. Effect Size
Report partial eta-squared for each factor:
$$\text{Partial } \eta^2 = \frac{SS_{effect}}{SS_{effect} + SS_{error}}$$

Interpretation:
- Small effect: 0.01
- Medium effect: 0.06
- Large effect: 0.14+

---

## Task 2: Interaction Analysis (1 point)

### Objective
Determine if k and max_depth interact in their effect on F1 score

### Methods

#### 1. Interaction Significance Test
From Two-Way ANOVA results, look at interaction p-value:
- If p < 0.05: Interaction is **statistically significant**
- If p > 0.05: Interaction is **not statistically significant**

#### 2. Interaction Plot
Create line plot showing interaction:
```python
# Plot 1: Lines for each k value, x-axis = max_depth
# Plot 2: Lines for each max_depth, x-axis = k

# Interpretation:
# - Parallel lines → No interaction
# - Non-parallel lines → Interaction present
# - Crossing lines → Strong interaction
```

#### 3. Simple Main Effects (if interaction significant)
If interaction exists, analyze effect of one factor at each level of the other:
- Effect of k at max_depth=3, 5, None
- Effect of max_depth at k=3, 5, 10

#### 4. Interpretation
Describe how the effect of k changes depending on max_depth (and vice versa)

### Results to Report
1. **Interaction Significance**
   - F-statistic and p-value
   - Effect size (partial η²)
   - Conclusion

2. **Nature of Interaction (if significant)**
   - Describe how factors interact
   - Which combination(s) perform best/worst
   - Practical implications

3. **Visualization**
   - Interaction plot with means and error bars
   - Show confidence intervals

### Example Interaction Findings
```
Example 1: No Interaction
- k=3, max_depth=3:     F1 = 0.55
- k=3, max_depth=5:     F1 = 0.62
- k=3, max_depth=None:  F1 = 0.65
- k=5, max_depth=3:     F1 = 0.56
- k=5, max_depth=5:     F1 = 0.63
- k=5, max_depth=None:  F1 = 0.66
→ max_depth effect is similar for all k values

Example 2: Interaction Present
- k=3, max_depth=3:     F1 = 0.55
- k=3, max_depth=5:     F1 = 0.58
- k=3, max_depth=None:  F1 = 0.60
- k=10, max_depth=3:    F1 = 0.62
- k=10, max_depth=5:    F1 = 0.67
- k=10, max_depth=None: F1 = 0.68
→ max_depth has larger effect at k=10 than at k=3
```

---

## Baseline Approach (`03_crfd_baseline.py`)

```python
# 1. Load preprocessed data from Phase 1

# 2. Create experimental grid
#    factors = {
#        'k': [3, 5, 10],
#        'max_depth': [3, 5, None]
#    }

# 3. For each combination of factors:
#    for repeat in range(10):
#        - Create StratifiedKFold(n_splits=k)
#        - Create RandomForestClassifier(max_depth=max_depth, random_state=seed)
#        - Perform cross-validation
#        - Collect F1 scores
#        - Store results

# 4. Organize results into DataFrame
#    Columns: [k, max_depth, repeat, fold, f1_score]
#    Save to crfd_results.csv

# 5. Aggregate results by (k, max_depth)
#    Calculate mean F1 for each combination

# 6. Perform Two-Way ANOVA
#    Test effects of k and max_depth
#    Print ANOVA table

# 7. Create interaction plot
#    Plot means with error bars
#    Show lines for each k value
#    Save plot

# 8. Report results
#    - Means and CIs for each factor
#    - ANOVA results
#    - Interaction summary
```

**Output:**
- `crfd_results.csv` - Raw results
- ANOVA table and interpretation
- Interaction plot

---

## Improved Approach (`03_crfd_improved.py`)

Enhancements:

### 1. Experimental Validation
- Verify stratified splits
- Check balance in combinations
- Validate fold sizes
- Ensure reproducibility with seed

### 2. ANOVA Assumptions
- Normality (Shapiro-Wilk test)
- Homogeneity of variance (Levene's test)
- Independence (already satisfied by design)
- Report: Do data meet assumptions?
- Alternative: Kruskal-Wallis if assumptions violated

### 3. Multiple Comparison Corrections
- Tukey HSD for pairwise comparisons
- Bonferroni correction if needed
- Report adjusted p-values

### 4. Effect Size
- Partial eta-squared
- Omega-squared
- Compare practical vs statistical significance

### 5. Comprehensive Visualization
- Interaction plots (both perspectives)
- Main effects plots
- Heatmap of mean F1 values
- Confidence interval bands

### 6. Model Diagnostics
- Residual plots
- Q-Q plots
- Homogeneity plots
- Check for outliers

---

## Analysis Script (`03_crfd_analysis.py`)

Dedicated analysis script:
- Load `crfd_results.csv`
- Perform comprehensive ANOVA
- Generate all visualizations
- Create detailed report
- Compare with CRD results

---

## Data Collection Strategy

### Pseudocode
```python
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

results = []
random_seed = 1234

# All factor combinations
k_values = [3, 5, 10]
max_depth_values = [3, 5, None]

for k, max_depth in product(k_values, max_depth_values):
    for repeat in range(10):
        # Create model
        model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=100,
            random_state=random_seed
        )
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=k, 
                              shuffle=True, 
                              random_state=random_seed + repeat)
        
        # Cross-validation
        f1_scores = cross_val_score(model, X, y, 
                                    cv=skf, 
                                    scoring='f1',
                                    n_jobs=-1)
        
        # Store results
        for fold_idx, f1 in enumerate(f1_scores):
            results.append({
                'k': k,
                'max_depth': max_depth,
                'repeat': repeat,
                'fold': fold_idx,
                'f1_score': f1
            })

df_results = pd.DataFrame(results)
df_results.to_csv('crfd_results.csv', index=False)
```

---

## Expected Results Table

Example factorial results:

| k   | max_depth | n   | Mean F1 | Std   | 95% CI        |
|-----|-----------|-----|---------|-------|---------------|
| 3   | 3         | 30  | 0.550   | 0.045 | [0.534-0.566] |
| 3   | 5         | 30  | 0.580   | 0.042 | [0.564-0.596] |
| 3   | None      | 30  | 0.600   | 0.048 | [0.583-0.617] |
| 5   | 3         | 30  | 0.562   | 0.043 | [0.546-0.578] |
| 5   | 5         | 30  | 0.592   | 0.041 | [0.576-0.608] |
| 5   | None      | 30  | 0.612   | 0.046 | [0.596-0.628] |
| 10  | 3         | 30  | 0.575   | 0.041 | [0.559-0.591] |
| 10  | 5         | 30  | 0.605   | 0.040 | [0.589-0.621] |
| 10  | None      | 30  | 0.625   | 0.044 | [0.609-0.641] |

---

## Output Files

```
Phase_3_CRFD/
├── README.md (this file)
├── 03_crfd_baseline.py
├── 03_crfd_improved.py
├── 03_crfd_analysis.py
├── crfd_results.csv                      # Raw results (required for submission)
└── outputs/
    ├── anova_table.txt
    ├── interaction_significance.txt
    ├── crfd_means_table.csv
    ├── main_effects_plot.png
    ├── interaction_plot_1.png             # Lines for each k
    ├── interaction_plot_2.png             # Lines for each max_depth
    ├── heatmap_means.png
    ├── residual_diagnostic_plots.png
    ├── crfd_comprehensive_report.txt
    └── crfd_detailed_analysis.pdf
```

---

## Submission Requirements

✓ `crfd_results.csv` with all 90 experimental results  
✓ Two-Way ANOVA table (k, max_depth, interaction effects)  
✓ Main effects analysis with means and 95% CIs  
✓ Interaction significance test with p-value  
✓ Interaction plot(s) clearly showing effects  
✓ Effect size (partial η²) for each factor  
✓ Conclusions on factor effects  
✓ Reproducible results with seed=1234  

---

## Run Instructions

```bash
# Baseline CRFD experiment
python 03_crfd_baseline.py

# Enhanced validation
python 03_crfd_improved.py

# Statistical analysis and visualization
python 03_crfd_analysis.py
```

---

## Important Notes

1. **Stratification:** Always use StratifiedKFold
2. **Replicates:** Need 10 replicates per treatment combination (not just 1)
3. **Random Seed:** Consistent seed usage for reproducibility
4. **Metrics:** F1 score with positive_class='yes'
5. **Assumptions:** Check and report ANOVA assumptions
6. **Interpretation:** Focus on both statistical AND practical significance

---

## Comparison with CRD Phase

| Aspect | CRD (Phase 2) | CRFD (Phase 3) |
|--------|-----------|---------|
| Factors | 1 (k only) | 2 (k and max_depth) |
| Levels | 3 | 3 × 3 = 9 |
| Experiments | 30 | 90 |
| Questions | Does k affect F1? | Do k and max_depth affect F1? Do they interact? |
| ANOVA | One-way | Two-way |

---

## Next Steps

→ Phase 4: Report Writing (Synthesize all findings)
