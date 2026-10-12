"""
FastAPI endpoints for video fact-checking.
Mounts under /api/video.
"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from video_processor import (
    VideoFactChecker, VideoAnalysisResult,
    _format_timestamp,
)

router = APIRouter(prefix="/api/video", tags=["video"])

# Store for video analysis results (in production, use Redis/DB)
_video_results: dict = {}


class VideoUploadResponse(BaseModel):
    """Response after uploading a video for analysis."""
    job_id: str
    status: str
    message: str


class VideoStatusResponse(BaseModel):
    """Status of a video analysis job."""
    job_id: str
    status: str  # "processing", "completed", "failed"
    progress: float = 0.0
    message: str = ""


class FactCheckItem(BaseModel):
    """Single fact-check result."""
    claim_id: str
    text: str
    speaker: str
    timestamp: str
    verdict: str
    confidence: float
    answer: str
    debate_triggered: bool
    conflict_type: Optional[str] = None
    sources: List[dict] = Field(default_factory=list)


class VideoAnalysisResponse(BaseModel):
    """Complete video analysis response."""
    job_id: str
    duration_seconds: float
    transcript: List[dict] = Field(default_factory=list)
    speakers: List[dict] = Field(default_factory=list)
    claims: List[dict] = Field(default_factory=list)
    fact_checks: List[FactCheckItem] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    error: Optional[str] = None


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    whisper_model: str = Form("base"),
    use_diarization: bool = Form(True),
    demo_mode: bool = Form(True),
):
    """
    Upload a video file for fact-checking analysis.

    The video will be processed to:
    1. Extract audio
    2. Transcribe speech
    3. Identify speakers (diarization)
    4. Extract factual claims
    5. Fact-check each claim using the MAD system
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    job_id = str(uuid.uuid4())[:8]

    # Save uploaded file
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_ext = Path(file.filename or "video.mp4").suffix
    file_path = upload_dir / f"{job_id}{file_ext}"

    try:
        content = await file.read()
        file_path.write_bytes(content)
        logger.info(f"Saved upload {job_id} ({len(content)} bytes) to {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Store job status
    _video_results[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "file_path": str(file_path),
        "whisper_model": whisper_model,
        "use_diarization": use_diarization,
        "demo_mode": demo_mode,
        "result": None,
    }

    # Start background processing
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(process_video_job, job_id)

    return VideoUploadResponse(
        job_id=job_id,
        status="processing",
        message="Video uploaded successfully. Use /api/video/status/{job_id} to check progress.",
    )


@router.get("/status/{job_id}", response_model=VideoStatusResponse)
async def get_status(job_id: str):
    """Check the status of a video analysis job."""
    if job_id not in _video_results:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _video_results[job_id]
    return VideoStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        message=job.get("message", ""),
    )


@router.get("/result/{job_id}", response_model=VideoAnalysisResponse)
async def get_result(job_id: str):
    """Get the complete analysis result for a video job."""
    if job_id not in _video_results:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _video_results[job_id]

    if job["status"] == "processing":
        raise HTTPException(status_code=202, detail="Analysis still in progress")

    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("message", "Analysis failed"))

    result = job.get("result")
    if result is None:
        raise HTTPException(status_code=500, detail="No result available")

    return _build_response(job_id, result)


@router.post("/analyze-sync", response_model=VideoAnalysisResponse)
async def analyze_video_sync(
    file: UploadFile = File(...),
    whisper_model: str = Form("base"),
    use_diarization: bool = Form(True),
    demo_mode: bool = Form(True),
):
    """
    Synchronous video analysis — processes the video and returns results immediately.
    May take several minutes for long videos.
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Save uploaded file
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_ext = Path(file.filename or "video.mp4").suffix
    file_path = upload_dir / f"{uuid.uuid4().hex[:8]}{file_ext}"

    try:
        content = await file.read()
        file_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Process video
    checker = VideoFactChecker(
        whisper_model=whisper_model,
        use_diarization=use_diarization,
    )

    result = checker.process(str(file_path))

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    # Fact-check claims
    if result.claims:
        from app import solve
        result = checker.fact_check_claims(result, solve)

    # Cleanup
    try:
        file_path.unlink()
    except Exception:
        pass

    return _build_response("sync", result)


def _build_response(job_id: str, result: VideoAnalysisResult) -> VideoAnalysisResponse:
    """Build API response from VideoAnalysisResult."""
    transcript_data = [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "speaker": seg.speaker,
            "timestamp": _format_timestamp(seg.start),
            "confidence": seg.confidence,
        }
        for seg in result.transcript
    ]

    speakers_data = [
        {
            "speaker_id": sp.speaker_id,
            "total_duration": sp.total_duration,
            "segment_count": sp.segment_count,
            "sample_text": sp.sample_text,
        }
        for sp in result.speakers
    ]

    claims_data = [
        {
            "claim_id": c.claim_id,
            "text": c.text,
            "speaker": c.speaker,
            "timestamp_start": c.timestamp_start,
            "timestamp_end": c.timestamp_end,
        }
        for c in result.claims
    ]

    fact_checks_data = [
        FactCheckItem(
            claim_id=fc.claim.claim_id,
            text=fc.claim.text,
            speaker=fc.claim.speaker,
            timestamp=_format_timestamp(fc.claim.timestamp_start),
            verdict=fc.verdict,
            confidence=fc.confidence,
            answer=fc.answer,
            debate_triggered=fc.debate_triggered,
            conflict_type=fc.conflict_type,
            sources=fc.sources,
        )
        for fc in result.fact_checks
    ]

    # Calculate metrics
    verdicts = [fc.verdict for fc in result.fact_checks]
    metrics = {
        "total_claims": len(result.claims),
        "total_speakers": len(result.speakers),
        "total_segments": len(result.transcript),
        "supported_count": verdicts.count("SUPPORTED") + verdicts.count("LIKELY_SUPPORTED"),
        "refuted_count": verdicts.count("REFUTED") + verdicts.count("DISPUTED"),
        "neutral_count": verdicts.count("NEUTRAL"),
        "unverifiable_count": verdicts.count("UNVERIFIABLE"),
        "debate_triggered_count": sum(1 for fc in result.fact_checks if fc.debate_triggered),
    }
    result.metrics = metrics

    return VideoAnalysisResponse(
        job_id=job_id,
        duration_seconds=result.duration_seconds,
        transcript=transcript_data,
        speakers=speakers_data,
        claims=claims_data,
        fact_checks=fact_checks_data,
        metrics=metrics,
        error=result.error,
    )


def process_video_job(job_id: str):
    """
    Background task to process a video job.
    Called by the worker/async handler.
    """
    if job_id not in _video_results:
        return

    job = _video_results[job_id]
    job["status"] = "processing"
    job["progress"] = 0.1

    try:
        checker = VideoFactChecker(
            whisper_model=job.get("whisper_model", "base"),
            use_diarization=job.get("use_diarization", True),
        )

        job["progress"] = 0.3
        result = checker.process(job["file_path"])

        if result.error:
            job["status"] = "failed"
            job["message"] = result.error
            return

        job["progress"] = 0.7

        # Fact-check claims
        if result.claims:
            from app import solve
            result = checker.fact_check_claims(result, solve)

        job["progress"] = 1.0
        job["status"] = "completed"
        job["result"] = result

    except Exception as e:
        logger.error(f"Video job {job_id} failed: {e}", exc_info=True)
        job["status"] = "failed"
        job["message"] = str(e)
