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
- Train RF tren train split stratified (mac dinh 80/20).
- Danh gia train + holdout + Stratified 5-fold CV (F1) tren tap train.
- `01_model_improved.py`
- Van giu max_depth=None, khong tuning.
- Bo sung holdout stratified 80/20, 95% CI cho CV.
- Bo sung classification report, ROC/PR curve, feature importance chuan hoa.
- Co ghi ca train metrics de theo doi muc do overfitting.
- `01_model_max_depth_comparison.py`
- So sanh nhieu gia tri `max_depth` tren cung train/holdout split va CV.
- Mac dinh so sanh: `3,5,7,10,None`.
- `01_model_analysis_lm.R`
- Phan tich ket qua CV F1 bang `lm()`.
- Co the bat `brm()` de so sanh (khong bat buoc) neu da cai dat package `brms`.
- `01_max_depth_lm_analysis.R`
- Phan tich `lm()` truc tiep cho so sanh nhieu `max_depth` (tu file `max_depth_comparison_cv_scores.csv`).
- Co bao cao mean + 95% CI theo tung muc `max_depth`, ANOVA, va tuy chon `brm()`.

## Cach chay
```bash
python Phase_1_Model/01_model_baseline.py
python Phase_1_Model/01_model_improved.py
```

Co the truyen tham so:
```bash
python Phase_1_Model/01_model_baseline.py --input-csv Phase_0_EDA/outputs/preprocessed_improved.csv --output-dir Phase_1_Model/outputs --random-seed 1234 --test-size 0.2
python Phase_1_Model/01_model_improved.py --input-csv Phase_0_EDA/outputs/preprocessed_improved.csv --output-dir Phase_1_Model/outputs --random-seed 1234 --test-size 0.2
python Phase_1_Model/01_model_max_depth_comparison.py --input-csv Phase_0_EDA/outputs/preprocessed_improved.csv --output-dir Phase_1_Model/outputs --random-seed 1234 --test-size 0.2 --max-depths 3,5,7,10,None
```

Phan tich thong ke bang R (chay trong RStudio hoac Rscript):
```bash
Rscript Phase_1_Model/01_model_analysis_lm.R
Rscript Phase_1_Model/01_model_analysis_lm.R --output_dir Phase_1_Model/outputs --use_brm TRUE --seed 1234
Rscript Phase_1_Model/01_max_depth_lm_analysis.R
Rscript Phase_1_Model/01_max_depth_lm_analysis.R --output_dir Phase_1_Model/outputs --use_brm TRUE --seed 1234
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
- `metrics_lm_baseline.txt`
- `metrics_lm_improved.txt`
- `metrics_lm_phase1.txt`
- `metrics_lm_phase1.json` (neu co package `jsonlite`)
- `lm_phase1_coefficients.csv`
- `lm_phase1_anova.csv` (chi co khi co ca baseline va improved)
- `lm_phase1_summary.txt`
- `phase1_cv_f1_boxplot.png`
- `brm_phase1_summary.txt` (neu bat `--use_brm TRUE` va co package `brms`)
- `brm_phase1_posterior_summary.csv` (neu bat `--use_brm TRUE`)
- `max_depth_lm_summary_by_group.csv`
- `lm_max_depth_coefficients.csv`
- `lm_max_depth_anova.csv`
- `lm_max_depth_report.txt`
- `lm_max_depth_metrics.json` (neu co package `jsonlite`)
- `lm_max_depth_boxplot.png`
- `brm_max_depth_summary.txt` (neu bat `--use_brm TRUE`)
- `brm_max_depth_posterior_summary.csv` (neu bat `--use_brm TRUE`)

### Max-depth comparison
- `max_depth_comparison_summary.csv`
- `max_depth_comparison_cv_scores.csv`
- `max_depth_comparison_plot.png`
- `max_depth_comparison.txt`
- `max_depth_comparison.json`

## Tieu chi chap nhan
- Chay duoc ca 2 lenh script baseline va improved.
- Co day du artifact nhu danh sach tren.
- CV dung 5 folds va F1 score nam trong [0, 1].
- Model luu ra co `max_depth=None` va `random_state=1234`.
- Chay lap lai cho ket qua nhat quan (khac biet neu co chi o muc lam tron hien thi).
