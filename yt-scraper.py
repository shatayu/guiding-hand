#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper-based YouTube Transcriber Pipeline (Improved)

Improvements in this version
- Single root output directory (configurable via --out-dir, default: ./run)
  with subfolders: cache/, logs/, transcripts/, artifacts/
- tqdm loading bars:
  • "Processing videos" bar for the URL list (ETA per video)
  • Per-video DOWNLOAD progress bar using yt-dlp progress hooks
  • Per-video TRANSCRIPTION progress bar with adaptive ETA based on video duration
- Much more verbose, timestamped logging for each step (download, metadata, cache, transcribe, checkpoint)
- All original features preserved:
  • Input via URLs or URL file
  • Robust audio download (yt-dlp → MP3)
  • Whisper transcription with per-segment timestamps
  • Rich YouTube metadata capture
  • Shelve cache (skip already-processed videos)
  • Crash protection & retries (tenacity)
  • Rotating file logs
  • Periodic checkpoints to PKL/CSV

Usage
    python yt-scraper.py \
        --urls https://youtu.be/9JI_EdiiG0c?si=aT48o_9avDdYmQtZ \
        --model small

    python yt-scraper.py \
        --url-file urls.txt \
        --model base \
        --language en \
        --out-dir run_whisper

"""
import argparse
import contextlib
import json
import logging
import os
import re
import shelve
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
import yt_dlp
import whisper


# ----------------------------
# Utilities
# ----------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sanitize_url(url: str) -> str:
    return url.strip().strip('"').strip("'")


def extract_video_id(url: str) -> Optional[str]:
    """
    Try to extract the 11-character YouTube ID.
    """
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


# ----------------------------
# Data Structures
# ----------------------------
@dataclass
class VideoRecord:
    uid: str
    url: str
    title: Optional[str]
    description: Optional[str]
    uploader: Optional[str]
    uploader_id: Optional[str]
    channel_id: Optional[str]
    channel: Optional[str]
    duration: Optional[int]
    upload_date: Optional[str]
    view_count: Optional[int]
    like_count: Optional[int]
    categories: Optional[List[str]]
    tags: Optional[List[str]]
    thumbnails: Optional[List[Dict]]
    webpage_url: Optional[str]
    audio_path: Optional[str]
    whisper_model: str
    language: Optional[str]
    transcript: List[Tuple[float, str]]  # [(start_time_seconds, text), ...]


# ----------------------------
# Paths & Logging (created later after arg parsing)
# ----------------------------
def setup_paths_and_logging(out_dir: Path) -> Dict[str, Path]:
    """
    Ensure a single root output directory with subdirs and configure logging.
    Returns a dict of important paths.
    """
    artifacts_dir = out_dir / "artifacts"
    audio_dir = out_dir / "transcripts"
    cache_dir = out_dir / "cache"
    logs_dir = out_dir / "logs"

    for d in (artifacts_dir, audio_dir, cache_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "pipeline.log"

    logger = logging.getLogger("yt_whisper_pipeline")
    logger.setLevel(logging.DEBUG)

    _fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_fmt)

    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_fmt)

    # Avoid duplicate handlers if rerun in same process
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.debug(f"[INIT] Output root: {out_dir}")
    logger.debug(f"[INIT] Subdirs: artifacts={artifacts_dir}, transcripts={audio_dir}, cache={cache_dir}, logs={logs_dir}")

    return {
        "ARTIFACTS_DIR": artifacts_dir,
        "AUDIO_DIR": audio_dir,
        "CACHE_DIR": cache_dir,
        "LOGS_DIR": logs_dir,
        "LOG_FILE": log_file,
    }


def make_ytdlp_opts(audio_dir: Path, download_bar: Optional[tqdm] = None) -> Dict:
    """
    Create yt-dlp options with a progress hook wired into a tqdm bar.
    """
    def _hook(d):
        if download_bar is None:
            return
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total and download_bar.total != total:
                download_bar.total = total
            # Set the bar to the current position
            download_bar.n = downloaded
            # Provide some extra info in the postfix
            spd = d.get("speed")
            eta = d.get("eta")
            if spd is not None and eta is not None:
                download_bar.set_postfix(speed=f"{spd/1e6:0.2f} MB/s", eta=f"{eta:0.0f}s")
            download_bar.refresh()
        elif status == "finished":
            # Mark complete
            if download_bar.total is not None:
                download_bar.n = download_bar.total
            download_bar.set_postfix_str("post-processing (ffmpeg)")
            download_bar.refresh()

    return {
        "quiet": True,
        "nocheckcertificate": True,
        "ignoreerrors": True,
        "noprogress": True,
        "retries": 3,
        "continuedl": True,
        "consoletitle": False,
        "outtmpl": str(audio_dir / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "progress_hooks": [_hook],
    }


# ----------------------------
# Download & Transcribe
# ----------------------------
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def ytdlp_download_and_info(url: str, ydl_opts: Dict, logger: logging.Logger) -> Dict:
    """
    Download audio as MP3 and return the info dict.
    Retries on transient errors.
    """
    logger.debug(f"[DL] Invoking yt-dlp for URL: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    logger.debug(f"[DL] yt-dlp completed for URL: {url}")
    return info


def mp3_path_for_video(audio_dir: Path, vid: str) -> Path:
    return audio_dir / f"{vid}.mp3"


def _transcribe_worker(model_name: str, audio_path: Path, language: Optional[str], result_holder: Dict, err_holder: Dict):
    """
    Worker to run whisper in a background thread so we can drive a tqdm progress bar concurrently.
    """
    try:
        model = whisper.load_model(model_name)
        # Set verbose=False to keep console tidy; our logs are descriptive anyway.
        result = model.transcribe(str(audio_path), language=language, verbose=False)
        result_holder["result"] = result
    except Exception as e:
        err_holder["error"] = e


def whisper_transcribe_with_progress(
    model_name: str,
    audio_path: Path,
    language: Optional[str],
    est_seconds: Optional[int],
    logger: logging.Logger,
) -> Dict:
    """
    Run Whisper with a progress bar that estimates time using the media duration.
    We run whisper in a background thread and update a tqdm bar off wall-clock time.

    The bar adapts if the job takes longer than the estimate.
    """
    # Provide a conservative default if no duration is known
    if not est_seconds or est_seconds <= 0:
        est_seconds = 60  # 1 minute default

    # Assume Whisper speed factor ~ 0.7x realtime on typical CPU/GPU setups.
    # You can tweak this factor to better fit your hardware.
    speed_factor = 0.7
    est_runtime = max(10, int(est_seconds / speed_factor))

    logger.info(f"[WHISPER] Starting transcription | model={model_name} | est_audio={est_seconds}s | est_runtime≈{est_runtime}s")

    result_holder: Dict = {}
    err_holder: Dict = {}
    t = threading.Thread(
        target=_transcribe_worker,
        args=(model_name, audio_path, language, result_holder, err_holder),
        daemon=True,
    )
    t.start()

    start = time.time()
    with tqdm(total=est_runtime, desc="Transcribing", unit="s", leave=True) as bar:
        while t.is_alive():
            elapsed = int(time.time() - start)
            # If we underestimated, extend the bar
            if elapsed > bar.total:
                bar.total = elapsed + 5  # extend by a small buffer
            bar.n = elapsed
            bar.set_postfix(elapsed=f"{elapsed}s")
            bar.refresh()
            time.sleep(0.2)

        # Ensure bar ends at final elapsed time
        final_elapsed = int(time.time() - start)
        if final_elapsed > bar.total:
            bar.total = final_elapsed
        bar.n = final_elapsed
        bar.set_postfix(elapsed=f"{final_elapsed}s", status="finalizing")
        bar.refresh()

    if "error" in err_holder:
        raise err_holder["error"]
    return result_holder["result"]


def segments_to_tuples(result: Dict) -> List[Tuple[float, str]]:
    tuples = []
    segments = result.get("segments", [])
    for seg in segments:
        start = float(seg.get("start", 0.0))
        text = seg.get("text", "").strip()
        if text:
            tuples.append((start, text))
    return tuples


# ----------------------------
# DataFrame Helpers
# ----------------------------
def build_dataframe(records: List[VideoRecord]) -> pd.DataFrame:
    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows)

    preferred = [
        "uid",
        "url",
        "title",
        "description",
        "uploader",
        "uploader_id",
        "channel",
        "channel_id",
        "upload_date",
        "duration",
        "view_count",
        "like_count",
        "categories",
        "tags",
        "audio_path",
        "whisper_model",
        "language",
        "transcript",
        "thumbnails",
        "webpage_url",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df.reindex(columns=cols)
    return df


def periodic_checkpoint(df: pd.DataFrame, artifacts_dir: Path, logger: logging.Logger) -> None:
    pkl_path = artifacts_dir / "transcripts.pkl"
    csv_path = artifacts_dir / "transcripts.csv"
    df.to_pickle(pkl_path)
    df.to_csv(csv_path, index=False)
    logger.info(f"[CHECKPOINT] Saved {pkl_path.name} & {csv_path.name} at {now_str()} (rows={len(df)})")


def validate_urls(urls: List[str]) -> List[str]:
    out = []
    for u in urls:
        u2 = sanitize_url(u)
        if u2:
            out.append(u2)
    return out


# ----------------------------
# Core per-video pipeline
# ----------------------------
def process_one(
    url: str,
    model_name: str,
    language: Optional[str],
    cache_db: shelve.Shelf,
    audio_dir: Path,
    artifacts_dir: Path,
    logger: logging.Logger,
) -> Optional[VideoRecord]:
    url = sanitize_url(url)
    if not url:
        logger.warning("[SKIP] Empty URL encountered.")
        return None

    logger.info(f"[BEGIN] Processing URL: {url}")

    # ID detection early for caching
    vid_guess = extract_video_id(url)
    if vid_guess:
        logger.debug(f"[ID] Extracted candidate video id: {vid_guess}")

    # Cache hit?
    if vid_guess and vid_guess in cache_db:
        logger.info(f"[CACHE] Hit for {vid_guess} → skipping download & transcription.")
        return cache_db[vid_guess]

    # --- Download & gather metadata (with tqdm bar) ---
    with tqdm(total=0, desc="Downloading", unit="B", unit_scale=True, leave=True) as dl_bar:
        ydl_opts = make_ytdlp_opts(audio_dir=audio_dir, download_bar=dl_bar)
        try:
            info = ytdlp_download_and_info(url, ydl_opts=ydl_opts, logger=logger)
        except Exception as e:
            logger.error(f"[ERROR] yt-dlp failed for {url}: {e}")
            return None

    vid = info.get("id") or vid_guess
    if not vid:
        logger.error(f"[ERROR] Could not determine video id for: {url}")
        return None
    logger.info(f"[META] Video id={vid} | title={info.get('title')} | duration={info.get('duration')}s")

    audio_mp3 = mp3_path_for_video(audio_dir, vid)
    if not audio_mp3.exists():
        # yt-dlp might have emitted a different ext; ensure MP3 path exists
        logger.debug("[AUDIO] Searching for emitted MP3 by id...")
        candidates = list(audio_dir.glob(f"{vid}.*"))
        for c in candidates:
            if c.suffix.lower() == ".mp3":
                audio_mp3 = c
                break

    if not audio_mp3.exists():
        logger.error(f"[ERROR] MP3 not found for video {vid}. Expected at: {audio_mp3}")
        return None
    logger.debug(f"[AUDIO] MP3 path resolved: {audio_mp3}")

    # --- Transcribe (with progress bar) ---
    try:
        duration = info.get("duration")
        result = whisper_transcribe_with_progress(
            model_name=model_name,
            audio_path=audio_mp3,
            language=language,
            est_seconds=duration,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"[ERROR] Whisper failed for {vid}: {e}")
        return None

    # Save raw whisper json (for audit)
    raw_json_path = audio_dir / f"{vid}.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"[SAVE] Raw whisper JSON saved: {raw_json_path.name}")

    # Convert to (timestamp, text) tuples
    transcript_tuples = segments_to_tuples(result)
    logger.info(f"[TRANSCRIPT] Segments captured: {len(transcript_tuples)}")

    # Build VideoRecord with rich metadata
    rec = VideoRecord(
        uid=vid,
        url=info.get("webpage_url") or url,
        title=info.get("title"),
        description=info.get("description"),
        uploader=info.get("uploader"),
        uploader_id=info.get("uploader_id"),
        channel_id=info.get("channel_id"),
        channel=info.get("channel"),
        duration=info.get("duration"),
        upload_date=info.get("upload_date"),
        view_count=info.get("view_count"),
        like_count=info.get("like_count"),
        categories=info.get("categories"),
        tags=info.get("tags"),
        thumbnails=info.get("thumbnails"),
        webpage_url=info.get("webpage_url"),
        audio_path=str(audio_mp3),
        whisper_model=model_name,
        language=language,
        transcript=transcript_tuples,
    )

    # Cache & return
    cache_db[vid] = rec
    cache_db.sync()
    logger.info(f"[CACHE] Stored {vid} at {now_str()}")

    logger.info(f"[END] Completed video id={vid}")
    return rec


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Whisper-based YouTube Transcriber Pipeline")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--urls", nargs="+", help="One or more YouTube URLs.")
    g.add_argument("--url-file", type=str, help="Path to a text file with one URL per line.")
    parser.add_argument("--model", type=str, default="small", help="Whisper model size (tiny, base, small, medium, large, etc.)")
    parser.add_argument("--language", type=str, default=None, help="Hint language code (e.g., 'en').")
    parser.add_argument("--resume", action="store_true", help="Resume mode (keeps previous cache & artifacts).")
    parser.add_argument("--cache-name", type=str, default="yt_cache", help="Base name for shelve cache (without extension).")
    parser.add_argument("--out-dir", type=str, default="run", help="Root output directory to hold logs/cache/transcripts/artifacts.")
    args = parser.parse_args()

    # Prepare paths & logging rooted under a single directory
    ROOT = Path(os.getcwd())
    OUT_DIR = (ROOT / args.out_dir).resolve()
    paths = setup_paths_and_logging(OUT_DIR)

    ARTIFACTS_DIR: Path = paths["ARTIFACTS_DIR"]
    AUDIO_DIR: Path = paths["AUDIO_DIR"]
    CACHE_DIR: Path = paths["CACHE_DIR"]
    logger = logging.getLogger("yt_whisper_pipeline")

    # Gather URLs
    if args.urls:
        urls = validate_urls(args.urls)
    else:
        with open(args.url_file, "r", encoding="utf-8") as f:
            urls = validate_urls([line for line in f if line.strip()])

    if not urls:
        logger.error("No valid URLs provided.")
        sys.exit(2)

    logger.info(f"[START] Total URLs to process: {len(urls)} | model={args.model} | language={args.language or 'auto'}")
    logger.info(f"[OUTPUT ROOT] {OUT_DIR}")

    cache_path = str(CACHE_DIR / args.cache_name)
    mode = "c" if args.resume else "n"  # new unless resume
    with contextlib.ExitStack() as stack:
        cache_db = stack.enter_context(shelve.open(cache_path, flag=mode, writeback=False))
        logger.info(f"[CACHE] Opened shelve at: {cache_path} (mode={'resume' if args.resume else 'new'})")

        records: List[VideoRecord] = []

        # If resuming and prior artifacts exist, seed DF to append to (by UID uniqueness)
        prior_df: Optional[pd.DataFrame] = None
        pkl_prior = ARTIFACTS_DIR / "transcripts.pkl"
        if args.resume and pkl_prior.exists():
            try:
                prior_df = pd.read_pickle(pkl_prior)
                logger.info(f"[RESUME] Loaded prior DataFrame with {len(prior_df)} rows.")
            except Exception as e:
                logger.warning(f"[RESUME] Failed reading prior pickle: {e}")

        # Process URLs with an outer tqdm
        with tqdm(total=len(urls), desc="Processing videos", unit="vid") as vids_bar:
            for url in urls:
                t0 = time.time()
                rec = process_one(
                    url=url,
                    model_name=args.model,
                    language=args.language,
                    cache_db=cache_db,
                    audio_dir=AUDIO_DIR,
                    artifacts_dir=ARTIFACTS_DIR,
                    logger=logger,
                )
                t1 = time.time()
                vids_bar.update(1)
                vids_bar.set_postfix(last_dur=f"{t1 - t0:0.1f}s")

                if rec:
                    records.append(rec)

                    # Crash protection: checkpoint after each success
                    new_df = build_dataframe(records)
                    if prior_df is not None:
                        combined = pd.concat([prior_df, new_df], ignore_index=True)
                        combined = combined.drop_duplicates(subset=["uid"], keep="last")
                        periodic_checkpoint(combined, artifacts_dir=ARTIFACTS_DIR, logger=logger)
                    else:
                        periodic_checkpoint(new_df, artifacts_dir=ARTIFACTS_DIR, logger=logger)

        # Final DF
        final_df = build_dataframe(records)
        if prior_df is not None:
            final_df = pd.concat([prior_df, final_df], ignore_index=True)
            final_df = final_df.drop_duplicates(subset=["uid"], keep="last")

        periodic_checkpoint(final_df, artifacts_dir=ARTIFACTS_DIR, logger=logger)

    logger.info(f"[DONE] Processed {len(final_df)} unique videos. Output root: {OUT_DIR}")


if __name__ == "__main__":
    main()
