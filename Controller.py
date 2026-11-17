#[1] 라우터
from fastapi import APIRouter
router = APIRouter( prefix="/api" )

# 서비스 객체 불러오기
from Service import body_service

# 스트리밍 응답 : 결과를 한 번에 return 하지 않고 여러 번 나눠서 보냄 (SSE)
from fastapi.responses import StreamingResponse

# 체지방률 예측 (스트리밍)
# http://127.0.0.1:8000/api/predict/stream
# {"age":27,"gender":"M","height_cm":172.3,"weight_kg":75.24,
#  "grip_force":54.9,"sit_and_bend":18.4,"sit_ups":60,"broad_jump":217}
@router.post("/predict/stream")
async def predict_stream( body:dict ) :
    return StreamingResponse( body_service.predict_stream( body ) , media_type="application/x-ndjson")