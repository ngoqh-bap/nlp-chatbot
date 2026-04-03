import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

import logging
import os
import uuid
from datetime import datetime, timezone
from collections import defaultdict
from time import time

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_cors_origins, get_cors_allow_credentials, get_log_level
from constants import Validation, ErrorMessage, SuccessMessage
from exceptions import ChatbotException, APIException, NLPException, DataException
from models import AdvancedChatRequest, ContextRequest, create_success_response
from services.nlp_service import get_nlp_service, merge_sticky_entities

log_dir = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_level = getattr(logging, get_log_level(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "chatbot.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="HUCE Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=get_cors_allow_credentials(),
    allow_methods=["*"],
    allow_headers=["*"],
)

request_counts = defaultdict(list)
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    current_time = time()

    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]

    if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "success": False,
                "error_code": "RATE_LIMIT_EXCEEDED",
                "error_message": "Quá nhiều yêu cầu. Vui lòng thử lại sau.",
                "retry_after": RATE_LIMIT_WINDOW,
            }
        )

    request_counts[client_ip].append(current_time)
    response = await call_next(request)
    return response


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def add_security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


logger.info("HUCE Chatbot API Server đang khởi động...")
nlp = get_nlp_service()
logger.info("NLP Service đã khởi tạo thành công")


@app.exception_handler(ChatbotException)
async def chatbot_exception_handler(request: Request, exc: ChatbotException):
    request_id = getattr(request.state, "request_id", None)
    logger.error(f"[{request_id}] ChatbotException: {exc.error_code} - {exc.message}")
    status_code = exc.status_code if isinstance(exc, APIException) else 422 if isinstance(exc, (NLPException,
                                                                                                DataException)) else 500
    return JSONResponse(status_code=status_code, content={
        "success": False, "error_code": exc.error_code, "error_message": exc.message,
        "details": exc.details, "request_id": request_id, "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(status_code=exc.status_code, content={
        "success": False, "error_code": f"HTTP_{exc.status_code}",
        "error_message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        "details": exc.detail if isinstance(exc.detail, dict) else {},
        "request_id": request_id, "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.error(f"[{request_id}] Unhandled exception: {type(exc).__name__} - {str(exc)}",
                 exc_info=True)
    return JSONResponse(status_code=500, content={
        "success": False, "error_code": "INTERNAL_SERVER_ERROR",
        "error_message": "Đã xảy ra lỗi không mong muốn. Vui lòng thử lại sau.",
        "details": {"debug_info": str(exc)} if logger.level == logging.DEBUG else {},
        "request_id": request_id, "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.get("/")
async def root():
    return create_success_response(message="HUCE Chatbot API đang hoạt động")


@app.get("/health")
async def health_check():
    try:
        nlp_status = "healthy" if nlp else "unhealthy"

        data_status = "healthy"
        try:
            from services.csv_service import CSVService
            csv_service = CSVService()
            majors = csv_service.get_majors()
            data_status = "healthy" if len(majors) > 0 else "no_data"
        except Exception as e:
            data_status = f"error: {str(e)}"

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {
                "nlp": nlp_status,
                "data": data_status,
            },
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
        )


@app.post("/chat/context")
async def manage_chat_context(req: ContextRequest):
    try:
        session_id = req.session_id or "default"
        if req.action == Validation.ACTION_GET:
            return create_success_response() | {"context": nlp.get_context(session_id)}
        elif req.action == Validation.ACTION_SET:
            context = req.context or {}
            nlp.set_context(session_id, context)
            return create_success_response(message=SuccessMessage.CONTEXT_UPDATED) | {
                "context": context}
        elif req.action == Validation.ACTION_RESET:
            nlp.reset_context(session_id)
            return create_success_response(message=SuccessMessage.CONTEXT_RESET)
        else:
            raise ValueError(ErrorMessage.INVALID_ACTION)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in /chat/context: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=ErrorMessage.INTERNAL_ERROR)


@app.post("/chat/advanced")
async def advanced_chat(req: AdvancedChatRequest):
    try:
        logger.info(
            f"/chat/advanced - Session: {req.session_id} - Message: {req.message[:100]}")
        session_id = req.session_id or "default"
        use_context = req.use_context if req.use_context is not None else True

        current_context = nlp.get_context(session_id) if use_context else {}
        result = nlp.handle_message(req.message, current_context)
        analysis, response = result["analysis"], result["response"]

        logger.info(
            f"/chat/advanced - Intent: {analysis['intent']} (score: {analysis['score']:.2f})")

        new_context = nlp.append_history(session_id,
                                         {"message": req.message, "intent": analysis["intent"],
                                          "response": response})
        current_intent = analysis["intent"]
        new_context["last_intent"] = current_intent

        current_entities = analysis["entities"]
        has_major = any(e.get('label') in ['TEN_NGANH', 'CHUYEN_NGANH', 'MA_NGANH'] for e in
                        current_entities)

        def get_intent_category(intent: str) -> str:
            categories = {"diem_chuan": "diem", "diem": "diem", "hoc_phi": "hoc_phi", "hoc_bong": "hoc_bong",
                          "nganh": "nganh_hoc", "chi_tieu": "chi_tieu", "phuong_thuc": "phuong_thuc",
                          "dieu_kien": "dieu_kien", "thoi_gian": "thoi_gian", "lich_trinh": "thoi_gian",
                          "to_hop": "to_hop", "khoi_thi": "to_hop"}
            for key, cat in categories.items():
                if key in intent:
                    return cat
            return "other"

        current_category = get_intent_category(current_intent)
        independent_categories = ["hoc_bong", "dieu_kien", "thoi_gian", "other"]

        if current_category in independent_categories:
            new_context["last_entities"] = current_entities
        elif has_major:
            new_context["last_entities"] = current_entities
        else:
            old_major = [e for e in current_context.get("last_entities", []) if
                         e.get('label') in ['TEN_NGANH', 'CHUYEN_NGANH', 'MA_NGANH']]
            new_context[
                "last_entities"] = old_major + current_entities

        turn = len(new_context.get("conversation_history") or [])
        merge_sticky_entities(new_context, current_entities, turn=turn)

        nlp.set_context(session_id, new_context)

        return {"analysis": analysis, "response": response, "context": new_context}

    except Exception as e:
        logger.error(f"Error in /chat/advanced: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "error": "Internal server error",
            "message": "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi của bạn. Vui lòng thử lại.",
        })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
