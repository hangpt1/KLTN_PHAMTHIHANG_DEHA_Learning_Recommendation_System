# Đóng góp cho Hệ thống Gợi ý Học tập (E-Learning Recommendation System)

Cảm ơn bạn đã quan tâm và muốn đóng góp cho dự án! Dưới đây là hướng dẫn chi tiết giúp bạn có thể tham gia phát triển hệ thống:

## 1. Fork & Clone Dự Án

- **Fork repository** trên GitHub về tài khoản của bạn.
- **Clone** bản fork đó về máy:
  ```bash
  git clone https://github.com/hangpt1/KLTN_PHAMTHIHANG_DEHA_Learning_Recommendation_System.git
  cd E-Learning-Recommendation-System
  ```

## 2. Cài đặt Môi trường (Local Setup)

Hệ thống được xây dựng bằng Python và Flask. Để chạy hệ thống ở local, vui lòng làm theo các bước sau:

```bash
# Tạo môi trường ảo (virtual environment)
python3 -m venv venv

# Kích hoạt môi trường ảo
# Trên macOS/Linux:
source venv/bin/activate
# Trên Windows:
# venv\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## 3. Chạy Ứng dụng & Kiểm thử (Testing)

```bash
# Khởi chạy hệ thống Flask
python app.py
```
Sau đó, hãy mở trình duyệt và truy cập vào: [http://localhost:5000](http://localhost:5000) để kiểm tra các tính năng của hệ thống (như hiển thị khoá học, mô hình gợi ý, kết quả bài test, v.v.).

## 4. Tạo Branch Mới

Luôn tạo một nhánh mới từ `main` trước khi thêm tính năng hoặc sửa lỗi:
```bash
# Tạo nhánh mới cho tính năng
git checkout -b feature/ten-tinh-nang-cua-ban

# Hoặc nhánh sửa lỗi
git checkout -b fix/mo-ta-loi
```

## 5. Quy tắc Viết Code (Code Style Guidelines)

- Tuân thủ tiêu chuẩn PEP 8 cho code Python.
- Đặc biệt với phần `src/` (chứa các thuật toán Recommendation, Feature Engineering): Hãy viết thêm docstring và comment giải thích cho các đoạn logic phức tạp hoặc các công thức học máy (machine learning) để người sau dễ bảo trì.

**Ví dụ Code Tốt:**
```python
def get_similar_courses(course_id, top_n=5):
    """
    Trả về top N các khoá học tương tự cho một course_id cụ thể.
    Dựa trên độ đo cosine similarity của nội dung khoá học.
    """
    course_features = extract_features(course_id)
    recommendations = similarity_model.predict(course_features)
    return recommendations[:top_n]
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
Nếu bạn tìm thấy lỗi, hãy tạo một Issue mới trên GitHub và đính kèm:
1. Tiêu đề rõ ràng.
2. Các bước tái hiện lỗi.
3. Kết quả mong muốn & kết quả thực tế.
4. Ảnh chụp màn hình (nếu có).


Cảm ơn bạn đã đóng góp! Những dòng code của bạn sẽ giúp hệ thống gợi ý này trở nên tuyệt vời hơn! 🙏
