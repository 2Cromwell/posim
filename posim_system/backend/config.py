"""
系统全局配置
"""
import sys
from pathlib import Path
from pydantic_settings import BaseSettings

# 将项目根目录加入 sys.path，以便导入 posim 核心模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 运行时数据目录
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """应用设置"""
    APP_NAME: str = "POSIM 舆情仿真推演系统"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # 数据库
    DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'posim_system.db'}"

    # CORS
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    # 上传文件大小限制 (bytes)
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB

    # 内置样例数据目录
    SAMPLE_DATA_DIRS: dict = {
        "tianjiaerhuan": str(PROJECT_ROOT / "scripts" / "tianjiaerhuan"),
        "wudatushuguan": str(PROJECT_ROOT / "scripts" / "wudatushuguan"),
        "xibeiyuzhicai": str(PROJECT_ROOT / "scripts" / "xibeiyuzhicai"),
    }

    class Config:
        env_prefix = "POSIM_"


settings = Settings()
