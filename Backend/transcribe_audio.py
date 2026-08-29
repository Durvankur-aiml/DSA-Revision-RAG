import os
import sys
import json
import subprocess
import whisper
import torch


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "data", "transcripts")
TEMP_WINDOW_DIR = os.path.join(BASE_DIR, "data", "audio", "_windows_temp")

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(TEMP_WINDOW_DIR, exist_ok=True)


WINDOW_DURATION = 300
CHUNK_DURATION = 90
GAP_WARNING_THRESHOLD = 5

# Per-video skip ranges (sponsor/promo segments), keyed by video_id.
# Add entries here as you identify them for each video.
SKIP_RANGES_BY_VIDEO = {
    "EAR7De6Goz4": [(170, 250)],   # C++ Basics L1 - Coding Ninjas sponsor
}


def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def is_in_skip_range(start, end, skip_ranges):
    for skip_start, skip_end in skip_ranges:
        if start < skip_end and end > skip_start:
            return True
    return False


def get_audio_duration(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    output = result.stdout.decode().strip()
    try:
        return float(output)
    except ValueError:
        raise RuntimeError(f"ffprobe returned unexpected output: {output}")


def split_audio_into_windows(path, total_dur, window_dur, out_dir):
    window_paths = []
    start = 0.0
    index = 0

    while start < total_dur:
        end = min(start + window_dur, total_dur)
        window_path = os.path.join(out_dir, f"window_{index:03d}.mp3")

        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-ss", str(start), "-to", str(end),
            "-c", "copy", window_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        window_paths.append({"path": window_path, "offset": start})
        start += window_dur
        index += 1

    return window_paths


def transcribe_one_video(audio_path, video_id, model, device):

    total_duration = get_audio_duration(audio_path)
    print(f"  Duration: {round(total_duration, 2)}s")

    windows = split_audio_into_windows(
        audio_path, total_duration, WINDOW_DURATION, TEMP_WINDOW_DIR
    )
    print(f"  Split into {len(windows)} window(s).")

    all_segments = []

    for i, window in enumerate(windows):
        print(f"    Window {i + 1}/{len(windows)} "
              f"(offset {round(window['offset'], 2)}s)...")

        try:
            result = model.transcribe(
                window["path"],
                fp16=(device == "cuda"),
                language="en",
                temperature=0,
                condition_on_previous_text=False,
                no_speech_threshold=0.3,
                logprob_threshold=-1.0
            )
        except Exception as e:
            print(f"    ERROR: Whisper failed on this window: {e}. Skipping window.")
            continue

        offset = window["offset"]
        for segment in result.get("segments", []):
            text = segment.get("text", "").strip()
            if not text:
                continue
            all_segments.append({
                "start": round(segment["start"] + offset, 2),
                "end": round(segment["end"] + offset, 2),
                "text": text
            })

    for window in windows:
        try:
            os.remove(window["path"])
        except OSError:
            pass

    # Gap validation (on raw segments)
    gaps_found = []
    for i in range(1, len(all_segments)):
        gap = all_segments[i]["start"] - all_segments[i - 1]["end"]
        if gap > GAP_WARNING_THRESHOLD:
            gaps_found.append({
                "previous_end": all_segments[i - 1]["end"],
                "current_start": all_segments[i]["start"],
                "gap_seconds": round(gap, 2)
            })

    # Skip-range filtering (sponsor/promo)
    skip_ranges = SKIP_RANGES_BY_VIDEO.get(video_id, [])
    filtered_segments = [
        s for s in all_segments
        if not is_in_skip_range(s["start"], s["end"], skip_ranges)
    ]
    skipped_count = len(all_segments) - len(filtered_segments)

    # Chunking
    chunks = []
    current_text = []
    chunk_start = None
    chunk_end = None

    for segment in filtered_segments:
        start, end, text = segment["start"], segment["end"], segment["text"]

        if chunk_start is None:
            chunk_start = start

        if current_text and (end - chunk_start > CHUNK_DURATION):
            chunks.append({
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2),
                "text": " ".join(current_text)
            })
            current_text = []
            chunk_start = start

        current_text.append(text)
        chunk_end = end

    if current_text:
        chunks.append({
            "start": round(chunk_start, 2),
            "end": round(chunk_end, 2),
            "text": " ".join(current_text)
        })

    return chunks, all_segments, gaps_found, skipped_count


def main():

    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("WARNING: GPU not detected. Using CPU.")

    if not os.path.isdir(AUDIO_DIR):
        fail(f"Audio directory does not exist: {AUDIO_DIR}")

    audio_files = [
        f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".mp3")
    ]

    if not audio_files:
        fail("No MP3 files found inside data/audio!")

    already_transcribed = {
        os.path.splitext(f)[0]
        for f in os.listdir(TRANSCRIPT_DIR)
        if f.lower().endswith(".json")
    }

    to_process = []
    skipped_files = []

    for f in audio_files:
        video_id = os.path.splitext(f)[0]
        if video_id in already_transcribed:
            skipped_files.append(f)
            continue
        to_process.append(f)

    if skipped_files:
        print(f"Skipping {len(skipped_files)} already-transcribed video(s).")

    if not to_process:
        print("\nAll audio files already transcribed. Nothing to do.")
        return

    print(f"\n{len(to_process)} new video(s) to transcribe:")
    for f in to_process:
        print(f"  - {f}")

    print("\nLoading Whisper SMALL model...")
    try:
        model = whisper.load_model("small", device=device)
    except Exception as e:
        fail(f"Failed to load Whisper model: {e}")

    success_count = 0
    failed_videos = []

    for f in to_process:

        video_id = os.path.splitext(f)[0]
        audio_path = os.path.join(AUDIO_DIR, f)

        print(f"\n{'=' * 50}")
        print(f"Transcribing: {f}")
        print("=" * 50)

        try:
            chunks, all_segments, gaps_found, skipped_count = transcribe_one_video(
                audio_path, video_id, model, device
            )
        except Exception as e:
            print(f"  ERROR: Failed to transcribe '{f}': {e}")
            failed_videos.append(f)
            continue

        if not chunks:
            print(f"  WARNING: No chunks produced for '{f}'. Skipping save.")
            failed_videos.append(f)
            continue

        output_file = os.path.join(TRANSCRIPT_DIR, f"{video_id}.json")
        temp_file = output_file + ".tmp"

        try:
            with open(temp_file, "w", encoding="utf-8") as file:
                json.dump(chunks, file, indent=4, ensure_ascii=False)
            os.replace(temp_file, output_file)
        except OSError as e:
            print(f"  ERROR: Failed to save transcript: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            failed_videos.append(f)
            continue

        print(f"\n  Saved {len(chunks)} chunks to: {output_file}")
        print(f"  Raw segments: {len(all_segments)}, "
              f"skipped (sponsor ranges): {skipped_count}")

        if gaps_found:
            print(f"  WARNING: {len(gaps_found)} gap(s) found:")
            for gap in gaps_found:
                print(f"    {gap['previous_end']}s -> {gap['current_start']}s "
                      f"(gap: {gap['gap_seconds']}s)")
        else:
            print("  No significant gaps detected.")

        success_count += 1

    try:
        os.rmdir(TEMP_WINDOW_DIR)
    except OSError:
        pass

    print(f"\n{'=' * 50}")
    print(f"BATCH COMPLETE — {success_count}/{len(to_process)} video(s) "
          f"transcribed successfully.")
    if failed_videos:
        print(f"Failed: {failed_videos}")
        print("Re-run this script to retry them.")
    print("=" * 50)


if __name__ == "__main__":
    main()