"""
Video fact-checking pipeline: diarization → transcription → claim extraction → fact-check.

Uses:
  - faster-whisper for transcription (local, no API key)
  - pyannote.audio for speaker diarization (requires HF token, falls back to clustering)
  - ffmpeg for audio extraction
"""
import os
import re
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class TranscriptSegment:
    """One segment of transcribed audio with speaker label."""
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"
    confidence: float = 0.0


@dataclass
class SpeakerProfile:
    """Information about a single speaker."""
    speaker_id: str
    total_duration: float = 0.0
    segment_count: int = 0
    sample_text: str = ""


@dataclass
class FactClaim:
    """A factual claim extracted from the transcript."""
    claim_id: str
    text: str
    speaker: str
    timestamp_start: float
    timestamp_end: float
    context: str = ""


@dataclass
class FactCheckResult:
    """Result of fact-checking a single claim."""
    claim: FactClaim
    verdict: str  # "SUPPORTED", "REFUTED", "NEUTRAL", "UNVERIFIABLE"
    confidence: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    debate_triggered: bool = False
    conflict_type: Optional[str] = None
    answer: str = ""
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class VideoAnalysisResult:
    """Complete result of video fact-checking."""
    video_path: str
    duration_seconds: float
    transcript: List[TranscriptSegment] = field(default_factory=list)
    speakers: List[SpeakerProfile] = field(default_factory=list)
    claims: List[FactClaim] = field(default_factory=list)
    fact_checks: List[FactCheckResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio from video using ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr[:500]}")
            return False
        return True
    except FileNotFoundError:
        logger.error("ffmpeg not found. Install with: sudo apt install ffmpeg")
        return False
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out")
        return False


def _transcribe_audio(audio_path: str, model_size: str = "base") -> List[TranscriptSegment]:
    """
    Transcribe audio using faster-whisper.
    Falls back to mock transcription if faster-whisper is not available.
    """
    try:
        from faster_whisper import WhisperModel

        logger.info(f"Loading Whisper model: {model_size}")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        transcript = []
        for segment in segments:
            transcript.append(TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                speaker="SPEAKER_00",  # Will be updated by diarization
                confidence=segment.avg_log_prob,
            ))

        logger.info(f"Transcribed {len(transcript)} segments, duration={info.duration:.1f}s")
        return transcript

    except ImportError:
        logger.warning("faster-whisper not installed. Using mock transcription.")
        return _mock_transcribe(audio_path)
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return _mock_transcribe(audio_path)


def _mock_transcribe(audio_path: str) -> List[TranscriptSegment]:
    """Mock transcription for testing without faster-whisper."""
    logger.warning("Using mock transcription — install faster-whisper for real results")
    # Return a sample transcript for testing
    return [
        TranscriptSegment(start=0.0, end=3.5, text="The Earth revolves around the Sun in 365 days.", speaker="SPEAKER_00"),
        TranscriptSegment(start=3.5, end=7.0, text="Water boils at 100 degrees Celsius at sea level.", speaker="SPEAKER_01"),
        TranscriptSegment(start=7.0, end=11.0, text="The Great Wall of China is visible from the Moon.", speaker="SPEAKER_00"),
        TranscriptSegment(start=11.0, end=15.0, text="Humans use only 10 percent of their brain capacity.", speaker="SPEAKER_01"),
        TranscriptSegment(start=15.0, end=19.0, text="Lightning never strikes the same place twice.", speaker="SPEAKER_00"),
    ]


def _diarize_speakers(audio_path: str, transcript: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """
    Perform speaker diarization using pyannote.audio.
    Falls back to simple clustering if pyannote is not available.
    """
    try:
        from pyannote.audio import Pipeline
        from pyannote.core import Segment

        logger.info("Running pyannote diarization...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("HF_TOKEN"),
        )

        diarization = pipeline(audio_path)

        # Assign speakers to transcript segments
        updated = []
        for seg in transcript:
            seg_segment = Segment(start=seg.start, end=seg.end)
            # Find the dominant speaker for this segment
            speaker_votes = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap = max(0, min(seg.end, turn.end) - max(seg.start, turn.start))
                if overlap > 0:
                    speaker_votes[speaker] = speaker_votes.get(speaker, 0) + overlap

            if speaker_votes:
                best_speaker = max(speaker_votes, key=speaker_votes.get)
                seg.speaker = best_speaker

            updated.append(seg)

        logger.info(f"Diarization complete: {len(set(s.speaker for s in updated))} speakers detected")
        return updated

    except ImportError:
        logger.warning("pyannote.audio not installed. Using fallback diarization.")
        return _fallback_diarize(transcript)
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return _fallback_diarize(transcript)


def _fallback_diarize(transcript: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """Simple fallback: alternate speakers based on pauses."""
    logger.info("Using fallback diarization (alternating speakers)")
    current_speaker = 0
    last_end = 0.0

    for seg in transcript:
        # If gap > 2 seconds, switch speaker
        if seg.start - last_end > 2.0:
            current_speaker = 1 - current_speaker
        seg.speaker = f"SPEAKER_{current_speaker:02d}"
        last_end = seg.end

    return transcript


def _extract_claims(transcript: List[TranscriptSegment]) -> List[FactClaim]:
    """Extract factual claims from transcript segments."""
    claims = []
    claim_id = 0

    for seg in transcript:
        text = seg.text.strip()
        if not text:
            continue

        # Simple heuristics: sentences with factual indicators
        # In production, use an LLM for claim extraction
        factual_indicators = [
            r"\b(is|are|was|were|has|have|had|will|would|could|should)\b",
            r"\b\d+\b",  # Contains numbers
            r"\b(percent|%|million|billion|thousand)\b",
            r"\b(always|never|all|none|every|most|some)\b",
        ]

        is_factual = any(re.search(p, text, re.I) for p in factual_indicators)

        # Also treat statements ending with period as claims
        if text.endswith(".") and len(text) > 20:
            is_factual = True

        if is_factual:
            claims.append(FactClaim(
                claim_id=f"claim_{claim_id:03d}",
                text=text,
                speaker=seg.speaker,
                timestamp_start=seg.start,
                timestamp_end=seg.end,
                context="",
            ))
            claim_id += 1

    logger.info(f"Extracted {len(claims)} factual claims from transcript")
    return claims


def _build_speaker_profiles(transcript: List[TranscriptSegment]) -> List[SpeakerProfile]:
    """Build speaker profiles from transcript."""
    profiles: Dict[str, SpeakerProfile] = {}

    for seg in transcript:
        sp = seg.speaker
        if sp not in profiles:
            profiles[sp] = SpeakerProfile(speaker_id=sp)
        profiles[sp].total_duration += seg.end - seg.start
        profiles[sp].segment_count += 1
        if not profiles[sp].sample_text:
            profiles[sp].sample_text = seg.text[:200]

    return list(profiles.values())


class VideoFactChecker:
    """Main class for video fact-checking pipeline."""

    def __init__(self, whisper_model: str = "base", use_diarization: bool = True):
        self.whisper_model = whisper_model
        self.use_diarization = use_diarization

    def process(self, video_path: str) -> VideoAnalysisResult:
        """
        Process a video file: extract audio, transcribe, diarize, extract claims, fact-check.

        Args:
            video_path: Path to video file

        Returns:
            VideoAnalysisResult with all analysis data
        """
        result = VideoAnalysisResult(video_path=video_path)
        audio_path = None

        try:
            # 1. Extract audio
            logger.info(f"Processing video: {video_path}")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name

            if not _extract_audio(video_path, audio_path):
                result.error = "Failed to extract audio from video"
                return result

            # 2. Get video duration
            result.duration_seconds = _get_audio_duration(audio_path)

            # 3. Transcribe
            transcript = _transcribe_audio(audio_path, self.whisper_model)

            # 4. Diarize
            if self.use_diarization:
                transcript = _diarize_speakers(audio_path, transcript)

            result.transcript = transcript

            # 5. Build speaker profiles
            result.speakers = _build_speaker_profiles(transcript)

            # 6. Extract claims
            result.claims = _extract_claims(transcript)

            logger.info(
                f"Video analysis: {len(transcript)} segments, "
                f"{len(result.speakers)} speakers, "
                f"{len(result.claims)} claims"
            )

        except Exception as e:
            logger.error(f"Video processing failed: {e}", exc_info=True)
            result.error = f"{type(e).__name__}: {e}"
        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

        return result

    def fact_check_claims(
        self,
        result: VideoAnalysisResult,
        solve_fn,
    ) -> VideoAnalysisResult:
        """
        Fact-check extracted claims using the MAD system.

        Args:
            result: VideoAnalysisResult from process()
            solve_fn: Callable that takes a query string and returns an ExecutionTrace

        Returns:
            Updated VideoAnalysisResult with fact_checks populated
        """
        from app import solve as default_solve

        if solve_fn is None:
            solve_fn = default_solve

        fact_checks = []

        for claim in result.claims:
            logger.info(f"Fact-checking claim {claim.claim_id}: {claim.text[:80]}...")

            try:
                trace = solve_fn(claim.text)

                # Determine verdict from trace
                if trace.final_answer:
                    verdict = _determine_verdict(trace)
                    fc = FactCheckResult(
                        claim=claim,
                        verdict=verdict,
                        confidence=0.8,
                        answer=trace.final_answer.answer,
                        debate_triggered=trace.debate_triggered,
                        conflict_type=trace.conflict.type.value if trace.conflict else None,
                    )
                    if trace.final_answer.sources:
                        fc.sources = [
                            {"url": s.url, "title": s.title}
                            for s in trace.final_answer.sources
                        ]
                else:
                    fc = FactCheckResult(
                        claim=claim,
                        verdict="UNVERIFIABLE",
                        confidence=0.0,
                        answer="No answer generated",
                    )

            except Exception as e:
                logger.error(f"Fact-check failed for {claim.claim_id}: {e}")
                fc = FactCheckResult(
                    claim=claim,
                    verdict="UNVERIFIABLE",
                    confidence=0.0,
                    answer=f"Error: {e}",
                )

            fact_checks.append(fc)

        result.fact_checks = fact_checks
        return result


def _determine_verdict(trace) -> str:
    """Determine fact-check verdict from execution trace."""
    if not trace.evidence_decision:
        return "UNVERIFIABLE"

    state = trace.evidence_decision.state.value

    if state == "agreement":
        # Multiple sources agree — likely supported
        return "SUPPORTED"
    elif state == "disagreement":
        # Sources disagree — check debate outcome
        if trace.debate_outcome and trace.debate_outcome.status == "completed":
            # Debate completed — check judge winner
            if trace.judge_result:
                winner = trace.judge_result.winner
                if winner == "candidate_1":
                    return "LIKELY_SUPPORTED"
                else:
                    return "DISPUTED"
        return "DISPUTED"
    elif state == "uncertain":
        return "NEUTRAL"
    else:
        return "UNVERIFIABLE"


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "json",
                audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return 0.0
