"""Local fleet prediction API and production React hosting."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from zipfile import BadZipFile

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .models.acv.predict import ACVPredictor
from .models.door.predict import DoorPredictor
from .models.rail.predict import ARTIFACT_VERSION, RailPredictor
from .models.shm.predict import SHMPredictor

LOGGER = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parents[1]


class BodyLimitMiddleware:
    """Bound multipart requests before their body is spooled to disk."""

    def __init__(self, app, maximum_bytes):
        self.app = app
        self.maximum_bytes = maximum_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/predict/"):
            return await self.app(scope, receive, send)
        limit = self.maximum_bytes + 1024 * 1024  # multipart overhead
        headers = dict(scope["headers"])
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = limit + 1
        if declared > limit:
            response = JSONResponse(
                {
                    "error": {
                        "code": "file_too_large",
                        "message": "Upload exceeds the file size limit.",
                    }
                },
                status_code=413,
            )
            return await response(scope, receive, send)
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(413, "Upload exceeds the file size limit.")
            return message

        await self.app(scope, bounded_receive, send)


def create_app(
    model_path: Path | None = None, maximum_bytes: int | None = None
) -> FastAPI:
    artifact = model_path or Path(
        os.environ.get(
            "RAIL_MODEL_PATH", str(APP_ROOT / "backend/artifacts/rail_pipeline.joblib")
        )
    )
    limit = (
        maximum_bytes
        if maximum_bytes is not None
        else int(os.environ.get("RAIL_MAX_FILE_BYTES", "67108864"))
    )
    inference_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(application):
        application.state.predictor = None
        try:
            application.state.predictor = RailPredictor.from_artifact(artifact)
        except Exception:
            LOGGER.exception("Rail model could not be loaded")
        application.state.predictors = {}
        for name, predictor_type in {
            "door": DoorPredictor,
            "acv": ACVPredictor,
            "shm": SHMPredictor,
        }.items():
            try:
                path = Path(
                    os.environ.get(
                        f"{name.upper()}_MODEL_PATH",
                        str(APP_ROOT / f"backend/artifacts/{name}_pipeline.joblib"),
                    )
                )
                application.state.predictors[name] = predictor_type.from_artifact(path)
            except Exception:
                LOGGER.exception("%s model could not be loaded", name)
                application.state.predictors[name] = None
        yield
        application.state.predictor = None
        application.state.predictors.clear()

    application = FastAPI(title="Fleet Diagnostic", lifespan=lifespan)
    application.add_middleware(BodyLimitMiddleware, maximum_bytes=limit)

    @application.exception_handler(HTTPException)
    async def http_error(request, exc):
        codes = {
            400: "invalid_request",
            413: "file_too_large",
            422: "invalid_recording",
            503: "model_unavailable",
            500: "inference_failed",
        }
        return JSONResponse(
            {
                "error": {
                    "code": codes.get(exc.status_code, "request_failed"),
                    "message": str(exc.detail),
                }
            },
            status_code=exc.status_code,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                {
                    "error": {
                        "code": "invalid_request",
                        "message": "Choose one recording file to upload.",
                    }
                },
                status_code=422,
            )
        return await request_validation_exception_handler(request, exc)

    @application.get("/api/health")
    def health():
        return {
            "status": "ready"
            if application.state.predictor is not None
            else "unavailable",
            "subsystems": {
                "rail": application.state.predictor is not None,
                **{
                    name: predictor is not None
                    for name, predictor in application.state.predictors.items()
                },
            },
            "model_version": getattr(
                application.state.predictor, "artifact_version", ARTIFACT_VERSION
            ),
            "maximum_file_bytes": limit,
        }

    @application.post("/api/predict/{subsystem}")
    async def predict(subsystem: str, file: UploadFile):
        try:
            if subsystem not in ("rail", "door", "acv", "shm"):
                raise HTTPException(404, "Unknown subsystem.")
            predictor = (
                application.state.predictor
                if subsystem == "rail"
                else application.state.predictors.get(subsystem)
            )
            extension = ".xlsx" if subsystem == "acv" else ".csv"
            if predictor is None:
                raise HTTPException(
                    503,
                    f"{subsystem.upper()} model is unavailable. Check the backend and model installation.",
                )
            filename = file.filename or ""
            if not filename.lower().endswith(extension) or any(
                c in filename for c in ("/", "\\", "\x00")
            ):
                raise HTTPException(
                    422,
                    f"Upload a {extension.upper()[1:]} recording with a plain filename.",
                )
            with tempfile.TemporaryDirectory(prefix=f"fleet-{subsystem}-") as directory:
                # Never use the user-controlled filename as a filesystem path.
                path = Path(directory) / f"recording{extension}"
                size = 0
                with path.open("wb") as target:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise HTTPException(
                                413,
                                f"File exceeds the {limit // (1024 * 1024)} MiB limit.",
                            )
                        target.write(chunk)
                if not size:
                    raise HTTPException(422, "The recording file is empty.")

                def infer():
                    with inference_lock:
                        if subsystem == "door":
                            return {
                                "cycles": [
                                    cycle.to_dict()
                                    for cycle in predictor.predict_stream(path)
                                ]
                            }
                        return predictor.predict_file(path).to_dict()

                try:
                    result = await run_in_threadpool(infer)
                except (ValueError, BadZipFile, EOFError) as exc:
                    raise HTTPException(422, str(exc)) from exc
                except Exception as exc:
                    LOGGER.exception("%s inference failed", subsystem)
                    raise HTTPException(
                        500,
                        "The recording could not be analysed. Retry or check the backend logs.",
                    ) from exc
                result["file_id"] = filename
                return {
                    "subsystem": subsystem,
                    "model_version": getattr(
                        predictor, "artifact_version", ARTIFACT_VERSION
                    )
                    if subsystem == "rail"
                    else f"{subsystem}-pipeline-v1",
                    **result,
                }
        finally:
            await file.close()

    dist = APP_ROOT / "frontend/dist"
    if (dist / "assets").is_dir():
        application.mount(
            "/assets", StaticFiles(directory=dist / "assets"), name="assets"
        )

    @application.get("/")
    @application.get("/rail")
    @application.get("/door")
    @application.get("/acv")
    @application.get("/shm")
    def frontend():
        if not (dist / "index.html").is_file():
            raise HTTPException(
                503,
                "Frontend is not built. Run npm ci and npm run build in app/frontend.",
            )
        return FileResponse(dist / "index.html")

    return application


app = create_app()
