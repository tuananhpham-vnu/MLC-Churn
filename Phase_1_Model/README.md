# Phase 1 - Mo hinh Random Forest cho du doan churn

## Muc tieu bai 1 (2 diem)
- Xay dung mo hinh Random Forest (RF) du doan `churn=yes/no`.
- Danh gia bang F1 score voi positive class la `yes` (tuong ung `churn_binary=1`).
- Su dung `random_seed=1234` de tai lap ket qua.
- Khong tinh chinh hyperparameter o Phase 1:
- `max_depth=None` (mac dinh), cac tham so khac de mac dinh cua sklearn.

## Dau vao va chinh sach feature
- Nguon du lieu mac dinh:
- `Phase_0_EDA/outputs/preprocessed_improved.csv`
- Bat buoc co cot muc tieu:
- `churn_binary`
- Chinh sach feature da khoa:
- Loai bo tat ca cot co tien to `area_code_`.
- Cac cot `*_charge` duoc xem la da loai tu Phase 0 (khong dua vao huan luyen).
- Giu cac cot `state_*` va cac bien hanh vi/chinh sach con lai.

## Cac script trong phase
- `01_model_baseline.py`
- Ban toi thieu dung de bai.
- Train RF tren toan bo du lieu sau khi loc feature.
- Danh gia train + Stratified 5-fold CV (F1).
- `01_model_improved.py`
- Van giu max_depth=None, khong tuning.
- Bo sung holdout stratified 80/20, 95% CI cho CV.
- Bo sung classification report, ROC/PR curve, feature importance chuan hoa.
- `01_model_analysis_lm.R`
- Phan tich ket qua CV F1 bang `lm()`.
- Co the bat `brm()` de so sanh (khong bat buoc) neu da cai dat package `brms`.

## Cach chay
```bash
python Phase_1_Model/01_model_baseline.py
python Phase_1_Model/01_model_improved.py
```

Co the truyen tham so:
```bash
python Phase_1_Model/01_model_baseline.py --input-csv Phase_0_EDA/outputs/preprocessed_improved.csv --output-dir Phase_1_Model/outputs --random-seed 1234
python Phase_1_Model/01_model_improved.py --input-csv Phase_0_EDA/outputs/preprocessed_improved.csv --output-dir Phase_1_Model/outputs --random-seed 1234
```

Phan tich thong ke bang R (chay trong RStudio hoac Rscript):
```bash
Rscript Phase_1_Model/01_model_analysis_lm.R
Rscript Phase_1_Model/01_model_analysis_lm.R --output_dir Phase_1_Model/outputs --use_brm TRUE --seed 1234
```

## Artifact dau ra
Tat ca file duoc luu trong `Phase_1_Model/outputs/` va tach biet bang hau to `baseline` / `improved`.

### Baseline
- `model_baseline.pkl`
- `metrics_baseline.json`
- `metrics_baseline.txt`
- `cv_f1_scores_baseline.csv`
- `feature_importances_baseline.csv`
- `feature_importance_baseline.png`
- `confusion_matrix_baseline.png`
- `selected_features_baseline.txt`

### Improved
- `model_improved.pkl`
- `metrics_improved.json`
- `metrics_improved.txt`
- `cv_f1_scores_improved.csv`
- `feature_importances_improved.csv`
- `feature_importance_improved.png`
- `confusion_matrix_improved.png`
- `classification_report_improved.json`
- `classification_report_improved.txt`
- `roc_curve_improved.png`
- `pr_curve_improved.png`
- `selected_features_improved.txt`

### R analysis (lm/brm)
- `phase1_cv_f1_combined.csv`
- `phase1_cv_f1_summary.csv`
- `lm_phase1_coefficients.csv`
- `lm_phase1_anova.csv` (chi co khi co ca baseline va improved)
- `lm_phase1_summary.txt`
- `phase1_cv_f1_boxplot.png`
- `brm_phase1_summary.txt` (neu bat `--use_brm TRUE` va co package `brms`)
- `brm_phase1_posterior_summary.csv` (neu bat `--use_brm TRUE`)

## Tieu chi chap nhan
- Chay duoc ca 2 lenh script baseline va improved.
- Co day du artifact nhu danh sach tren.
- CV dung 5 folds va F1 score nam trong [0, 1].
- Model luu ra co `max_depth=None` va `random_state=1234`.
- Chay lap lai cho ket qua nhat quan (khac biet neu co chi o muc lam tron hien thi).
