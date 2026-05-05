
## 1. Fork & Clone Dự Án
- **Fork repository** trên GitHub về tài khoản của bạn.
- **Clone** bản fork đó về máy:
  ```bash
  git clone https://github.com/hangpt1/KLTN_PHAMTHIHANG_DEHA_Learning_Recommendation_System.git
  cd E-Learning-Recommendation-System
  ```
## 2. Cài đặt Môi trường (Local Setup)
Hệ thống được xây dựng bằng Python và Flask. Để chạy hệ thống ở local, vui lòng làm theo các bước sau:
Vào đúng thư mục dự án
cd "/Users/phamthihang/E-Learning-Recommendation-System"

2) Chạy đánh giá Recommender (K-Fold 10 cổ điển)
python3 run_kfold_evaluation.py
Kết quả chính sẽ nằm ở:
* evaluation_results/kfold_summary.csv
* evaluation_results/kfold_detailed_results.txt

3) Chạy đánh giá Grade Predictor + K-Means
python3 run_ml_evaluation.py
Kết quả lưu tại:
* evaluation_results/ml_components_report.txt

4) Chạy demo gợi ý cho 1 người dùng (ví dụ S001)
python3 src/recommendation_engine.py

Script này sẽ in:
* Top-10 course gợi ý cho S001
* Learning path mẫu
* Similar courses mẫu

5) Chạy web app để demo giao diện
python3 app.py
Sau đó mở trình duyệt:
* http://127.0.0.1:5001
Bạn có thể đăng nhập bằng mã học viên demo (ví dụ S001) để xem dashboard, gợi ý, analytics.

6) (Tuỳ chọn) Demo nhanh dự đoán điểm bằng script 1 dòng
python3 -c "from src.ml_features import run_ml_demo; run_ml_demo('data')"

Lệnh này chạy demo ML tổng quát (grade predictor + clustering) ở mức script.

7) (Tuỳ chọn) Lưu log ra file để nộp/chiếu
python3 run_kfold_evaluation.py | tee evaluation_results/demo_kfold_stdout.txt
python3 run_ml_evaluation.py | tee evaluation_results/demo_ml_stdout.txt

Sau đó, hãy mở trình duyệt và truy cập vào: [http://localhost:5000](http://localhost:5000) để kiểm tra các tính năng của hệ thống (như hiển thị khoá học, mô hình gợi ý, kết quả bài test, v.v.).
## 4. Tạo Branch Mới
Luôn tạo một nhánh mới từ `main` trước khi thêm tính năng hoặc sửa lỗi:
```bash
# Tạo nhánh mới cho tính năng
git checkout -b feature/ten-tinh-nang-cua-ban

# Hoặc nhánh sửa lỗi
git checkout -b fix/mo-ta-loi
```

## 6. Commit & Push

Viết commit message rõ ràng:
```bash
# Ví dụ commit chuẩn:
git commit -m "feat: Thêm mô hình Content-Based Filtering cho gợi ý khoá học"
git commit -m "fix: Sửa lỗi tính toán sai điểm dự đoán trong Grade Predictor"
git commit -m "docs: Cập nhật tài liệu hướng dẫn cài đặt"
```
Sau đó push lên bản fork của bạn:
```bash
git push origin feature/ten-tinh-nang-cua-ban
```