# Datathon 2026: The Gridbreaker - Sales Forecasting
**Tên đội thi:** Dataminator

## 1. Cấu trúc thư mục
Để chạy được mô hình, Ban giám khảo vui lòng đặt các file dữ liệu (`sales.csv`, `sales_test.csv`...) cùng cấp với file code chính.

```text
📦 Datathon2026_TheGridbreaker
 ┣ 📜 Model.py                   # Source code chính chứa Pipeline huấn luyện & Dự báo
 ┣ 📜 Submission.csv             # Kết quả dự báo
 ┣ 📜 shap_executive_view.png    # Biểu đồ SHAP Explainability
 ┗ 📜 README.md                  # Hướng dẫn chạy code
```

## 2. Các thư viện cần thiết
Mô hình yêu cầu các thư viện sau:
- `pandas`, `numpy` (Xử lý dữ liệu)
- `catboost`, `lightgbm` (Huấn luyện mô hình Gradient Boosting)
- `shap` (Trích xuất độ quan trọng của đặc trưng - Explainability)
- `matplotlib`, `seaborn` (Trực quan hóa)

## 3. Hướng dẫn chạy lại kết quả
Mô hình đã được cố định random seed (`SEED = 42`) để đảm bảo tính tái lập 100%.
1. Đảm bảo toàn bộ các file `.csv` của ban tổ chức (đặc biệt là `sales.csv` và `sample_submission.csv`) nằm cùng thư mục với script.
2. Chạy trực tiếp file `V66_Ultimate_Synthesis.py`.
3. Hệ thống sẽ tự động thực hiện Feature Engineering, Train mô hình (CatBoost + LightGBM), áp dụng bộ lọc COGS [70% - 95%] và xuất ra file nộp bài định dạng `.csv`. Đồng thời, hệ thống sẽ tự động trích xuất biểu đồ `shap_executive_view.png`.
