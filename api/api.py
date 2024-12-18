from fastapi import FastAPI, APIRouter
from api_router import ops_router
from api_router import tasks_router

router = APIRouter()
router.include_router(ops_router.router)
router.include_router(tasks_router.router)

app = FastAPI(title='Speech AI', version='1.0.0')
app.include_router(router)