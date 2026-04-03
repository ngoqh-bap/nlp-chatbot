# HUCE Chatbot - Hệ Thống Chatbot Tuyển Sinh

> Chatbot tra cứu thông tin tuyển sinh Đại học Xây dựng Hà Nội với NLP tiếng Việt

## 🎯 Tính Năng

### Tra Cứu Thông Tin

- ✅ Điểm chuẩn các ngành theo năm
- ✅ Học phí và học bổng
- ✅ Thông tin ngành học, tổ hợp môn
- ✅ Phương thức xét tuyển, lịch tuyển sinh
- ✅ Thông tin liên hệ

### NLP Tiếng Việt

- ✅ Intent detection (legacy TF‑IDF + softmax probabilities; optional `transformers` path planned)
- ✅ Entity extraction (pattern + dictionary + spans)
- ✅ Context management (nhớ 10 câu hỏi gần nhất)
- ✅ Fallback thông minh khi không hiểu

### Bảo Mật

- ✅ Input sanitization (XSS/SQLi)
- ✅ Rate limiting
- ✅ Request validation
- ✅ Error handling

---

## 🛠 Công Nghệ

### Backend

- **FastAPI** - Web framework
- **Transformers + PyTorch** - optional model-based NLP
- **Deterministic tokenization** - Vietnamese text normalization
- **Pydantic** - Data validation
- **pytest** - Testing framework
- **pandas** - CSV processing

### Frontend

- **Reflex** - Python web framework
- **WebSocket** - Real-time communication

### Data

- **13 CSV files** - Admission data
- **Caching** - Optimized with mtime checking

---

## 📁 Cấu Trúc Dự Án

```
DATN/
├── main.py                 # FastAPI application
├── models.py               # Pydantic models
├── config.py              # Configuration
├── constants.py           # Constants
│
├── nlu/                   # NLP Pipeline
│   ├── pipeline.py        # Orchestration
│   ├── intent.py          # Intent detection
│   ├── entities.py        # Entity extraction
│   └── preprocess.py      # Text preprocessing
│
├── services/              # Business Logic
│   ├── nlp_service.py     # NLP facade
│   ├── csv_service.py     # Data loading
│   ├── handlers/          # Intent handlers
│   └── processors/        # Data processors
│
├── exceptions/            # Custom Exceptions
│   ├── nlp_exceptions.py
│   ├── data_exceptions.py
│   └── api_exceptions.py
│
├── utils/                 # Utilities
│   └── sanitize.py        # Input sanitization
│
├── tests/                 # Test Suite
│   ├── unit/              # Unit tests (122)
│   └── integration/       # Integration tests (10)
│
├── data/                  # CSV Data
│   ├── admission_scores.csv
│   ├── majors.csv
│   ├── tuition.csv
│   └── ...
│
└── frontend/              # Reflex Frontend
    └── chatbot/
```

---

## 🚀 Bắt Đầu

### Yêu Cầu

- Python 3.13+
- uv package manager
- Git

### Cài Đặt

```bash
# 1. Clone repository
git clone https://github.com/your-org/huce-chatbot.git
cd huce-chatbot

# 2. Cài đặt dependencies
pip install uv
uv sync

# 3. Cấu hình environment (tùy chọn)
cp env.example .env
# Chỉnh sửa .env nếu cần

# 4. Chạy tests để verify
pytest

# 5. Chạy backend
uvicorn main:app --reload

# 6. Chạy frontend (terminal khác)
cd frontend
reflex run
```

### Truy Cập

- **Backend API:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs
- **Frontend:** http://localhost:3000

---

## 🧪 Testing

### Chạy Tests

```bash
# Chạy tất cả tests
pytest

# Chạy với coverage
pytest --cov=. --cov-report=html

# Chạy tests cụ thể
pytest tests/unit/
pytest tests/integration/
```

---

## 📡 API Endpoints

### 1. Health Check

```bash
GET /
```

### 2. Chat với NLP

```bash
POST /chat/advanced
{
  "message": "Điểm chuẩn ngành Kiến trúc?",
  "session_id": "user_123",
  "use_context": true
}
```

### 3. Quản Lý Context

```bash
POST /chat/context
{
  "action": "get|set|reset",
  "session_id": "user_123"
}
```

Chi tiết: [API_GUIDE.md](./API_GUIDE.md)

---

## 🔒 Bảo Mật

### Input Sanitization

- ✅ XSS prevention (HTML escaping)
- ✅ SQL injection prevention (pattern removal)
- ✅ Spam detection (multiple heuristics)
- ✅ Length limits (prevent abuse)
- ✅ Session validation

### Error Handling

- ✅ 15 custom exception types
- ✅ Standardized error responses
- ✅ Request ID tracking
- ✅ No stack traces in production

---

## 📈 Roadmap

### ✅ Đã Hoàn Thành

- [x] Core NLP pipeline
- [x] Context management
- [x] 132 tests với 100% pass rate
- [x] Exception handling
- [x] Input sanitization
- [x] Complete documentation

### 🔄 Đang Phát Triển

- [ ] Rate limiting
- [ ] Authentication (API key)
- [ ] Monitoring dashboard

### 📅 Tương Lai

- [ ] Database migration (CSV → PostgreSQL)
- [ ] Custom NER model training
- [ ] Personalized responses
- [ ] Multi-language support

---

## 🤝 Đóng Góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng đọc:

1. [CONTRIBUTING.md](./CONTRIBUTING.md) - Hướng dẫn đóng góp
2. [ARCHITECTURE.md](./ARCHITECTURE.md) - Hiểu kiến trúc
3. [TESTING_GUIDE.md](./TESTING_GUIDE.md) - Viết tests

### Quy Trình

```bash
# 1. Fork repository
# 2. Tạo branch
git checkout -b feature/your-feature

# 3. Code và test
pytest

# 4. Commit với message rõ ràng
git commit -m "feat: add new feature"

# 5. Push và tạo PR
git push origin feature/your-feature
```

### Tài Nguyên

- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **GitHub:** [Link to repository]
- **Wiki:** [Link to wiki]

---

## 🌟 Tính Năng Nổi Bật

### 1. Smart Context Management

Tự động hiểu câu hỏi tiếp theo mà không cần nhắc lại ngành học:

```
User: "Điểm chuẩn ngành CNTT?"
Bot:  "Điểm chuẩn CNTT là 25.5..."

User: "Còn học phí thế nào?"
Bot:  "Học phí ngành CNTT là 31 triệu/năm"
      ↑ Tự động hiểu đang hỏi về CNTT
```

### 2. Comprehensive Testing

- 132 tests cover all critical paths
- 100% pass rate maintained
- Sub-second execution time
- CI-ready infrastructure

### 3. Production-Ready

- Exception handling cho mọi error case
- Request ID tracking cho debugging
- Input sanitization cho security
- Comprehensive documentation
