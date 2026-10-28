"""
Extract 1 frame per second from each video (into data/frames_1fps/).

Goes through the videos of the annotated matches in the manifest (train, val and test). Each video gets a folder
with its frames: 00000.jpg is second 0, 00001.jpg is second 1, and so on.
Videos that already have their frames are skipped.

Usage (from the repo root):  python events/extract_frames.py        (~5-10 minutes per video)

Two tricks so it runs faster and doesn't need watching (it is a long step):
  1. It processes 3 videos at a time (each with its own ffmpeg).
  2. If a video is missing frames at the end, it retries on its own (up to 3 times).
"""
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import config

VIDEOS_AT_ONCE = 3         # trick 1
ATTEMPTS = 3               # trick 2


def duration_seconds(video):
    """Asks ffprobe how long the video is."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_frames(video, output_folder):
    """Takes 1 frame per second from the video with ffmpeg.

    Watch out for round=up: with the plain fps=1 filter, frame N shows second N + 0.5
    (ffmpeg rounds to the nearest second and keeps the last frame). With round=up
    frame N shows exactly second N. We checked this by comparing images.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error",   # -nostdin: don't read the terminal's keyboard
        "-i", str(video),
        "-vf", "fps=1:round=up",        # 1 frame per second, frame N = second N
        "-q:v", "2",                    # high JPG quality (2 is almost lossless)
        "-start_number", "0",           # the first file is 00000.jpg (second 0)
        str(output_folder / "%05d.jpg"),
    ]
    subprocess.run(command)


def process_video(video):
    """Extracts the frames of one video and checks they are all there. Returns a message."""
    name = video.stem
    folder = config.FRAMES_1FPS / name

    if folder.exists() and any(folder.iterdir()):
        count = len(list(folder.glob("*.jpg")))
        return f"{name}: already done ({count} frames), skipping"

    expected = int(duration_seconds(video)) + 1       # one frame per second of video
    for attempt in range(1, ATTEMPTS + 1):
        extract_frames(video, folder)
        # Check: there must be one frame per second (give or take 1). Sometimes ffmpeg
        # stops halfway without warning; so we count and, if frames are missing, retry.
        count = len(list(folder.glob("*.jpg")))
        if abs(count - expected) <= 2:
            return f"{name}: done ({count} frames)"
        print(f"  {name}: got {count} frames, expected {expected}; retrying ({attempt} of {ATTEMPTS})")
        shutil.rmtree(folder)                             # delete the incomplete frames and start over
    return f"{name}: ERROR, could not extract all frames after {ATTEMPTS} attempts"


def main():
    videos = sorted(config.VIDEOS.glob("*.mp4")) + sorted(config.VIDEOS.glob("*.mov"))
    videos = [v for v in videos if v.stem in config.TRAIN + config.VAL + config.TEST]      # the annotated ones
    print(f"Processing {len(videos)} videos, {VIDEOS_AT_ONCE} at a time\n")

    # Trick 1: the pool hands the videos out to 3 workers; when one finishes, it takes the next.
    # Messages come out in list order: if a video finishes before the previous one,
    # its message waits for the previous one.
    done = 0
    with ThreadPoolExecutor(VIDEOS_AT_ONCE) as pool:
        for message in pool.map(process_video, videos):
            done += 1
            print(f"[{done} of {len(videos)}] {message}")

    print("\nAll videos done.")


if __name__ == "__main__":
    main()
