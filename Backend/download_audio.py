import os
import sys
import json

import yt_dlp


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
METADATA_DIR = os.path.join(BASE_DIR, "data", "metadata")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def already_downloaded(video_id):
    mp3_path = os.path.join(AUDIO_DIR, f"{video_id}.mp3")
    metadata_path = os.path.join(METADATA_DIR, f"{video_id}.json")
    return os.path.exists(mp3_path) and os.path.exists(metadata_path)


def save_metadata(video_id, video_title):
    metadata = {
        "video_id": video_id,
        "video_title": video_title,
        "youtube_url": f"https://youtu.be/{video_id}",
        "audio_filename": f"{video_id}.mp3",
    }
    metadata_path = os.path.join(METADATA_DIR, f"{video_id}.json")

    try:
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
        return True
    except OSError as e:
        print(f"    ERROR: Failed to save metadata for {video_id}: {e}")
        return False


def download_one_video(video_id):

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(AUDIO_DIR, f"{video_id}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    video_url = f"https://youtu.be/{video_id}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        print(f"    ERROR: Download failed for {video_id}: {e}")
        return False

    expected_mp3 = os.path.join(AUDIO_DIR, f"{video_id}.mp3")

    if not os.path.exists(expected_mp3):
        print(f"    ERROR: Download reported OK but MP3 not found: "
              f"{expected_mp3}")
        return False

    return True


def get_int_input(prompt, default):
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  Invalid number, using default: {default}")
        return default


def main():

    playlist_url = input("Enter YouTube playlist URL: ").strip()

    if not playlist_url:
        fail("No URL entered.")

    print("\nHow many videos do you want to process in this batch?")
    batch_size = get_int_input(
        "  Batch size (press Enter for default 10): ", 10
    )

    print("\nWhere should this batch start in the playlist? "
          "(1 = beginning)")
    start_position = get_int_input(
        "  Start position (press Enter for default 1): ", 1
    )

    if batch_size < 1:
        fail("Batch size must be at least 1.")
    if start_position < 1:
        fail("Start position must be at least 1.")

    # ==========================================
    # STEP 1 — FETCH PLAYLIST ENTRY LIST
    # ==========================================

    print("\nFetching playlist info...")

    list_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(list_opts) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
    except Exception as e:
        fail(f"Failed to fetch playlist info: {e}")

    entries = playlist_info.get("entries")

    if not entries:
        fail("No videos found in this playlist. Check the URL.")

    entries = [e for e in entries if e]
    total_in_playlist = len(entries)

    print(f"Playlist: {playlist_info.get('title', 'Unknown')}")
    print(f"Total videos in playlist: {total_in_playlist}")

    # ==========================================
    # STEP 2 — SLICE OUT THE REQUESTED BATCH
    # ==========================================

    start_index = start_position - 1
    end_index = start_index + batch_size

    if start_index >= total_in_playlist:
        fail(f"Start position {start_position} is beyond the playlist "
             f"length ({total_in_playlist}).")

    batch_entries = entries[start_index:end_index]

    print(f"\nThis run will process videos "
          f"{start_position} to "
          f"{min(start_index + len(batch_entries), total_in_playlist)} "
          f"({len(batch_entries)} video(s)).")

    # ==========================================
    # STEP 3 — DOWNLOAD EACH VIDEO IN THE BATCH
    # ==========================================

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    failed_videos = []

    for offset, entry in enumerate(batch_entries):

        playlist_position = start_position + offset
        video_id = entry.get("id")
        video_title = entry.get("title", "Unknown title")

        if not video_id:
            print(f"[{playlist_position}] Skipping entry with no video_id.")
            failed_count += 1
            continue

        print(f"\n[{playlist_position}/{total_in_playlist}] "
              f"{video_title} ({video_id})")

        if already_downloaded(video_id):
            print(f"    Already downloaded. Skipping.")
            skipped_count += 1
            continue

        print(f"    Downloading...")

        success = download_one_video(video_id)

        if not success:
            failed_count += 1
            failed_videos.append({"video_id": video_id, "title": video_title})
            continue

        metadata_ok = save_metadata(video_id, video_title)

        if not metadata_ok:
            failed_count += 1
            failed_videos.append({"video_id": video_id, "title": video_title})
            continue

        downloaded_count += 1
        print(f"    Done.")

    # ==========================================
    # FINAL SUMMARY
    # ==========================================

    print("\n" + "=" * 50)
    print("BATCH DOWNLOAD COMPLETE")
    print("=" * 50)
    print(f"Batch: videos {start_position} to "
          f"{start_position + len(batch_entries) - 1} "
          f"of {total_in_playlist}")
    print(f"Newly downloaded: {downloaded_count}")
    print(f"Already had (skipped): {skipped_count}")
    print(f"Failed: {failed_count}")

    if failed_videos:
        print(f"\nFailed videos (re-run this same batch to retry):")
        for fv in failed_videos:
            print(f"  - {fv['title']} ({fv['video_id']})")

    next_start = start_position + batch_size
    if next_start <= total_in_playlist:
        print(f"\nNext batch: run again with start position {next_start} "
              f"to continue.")
    else:
        print(f"\nThis was the last batch — playlist fully covered.")


if __name__ == "__main__":
    main()