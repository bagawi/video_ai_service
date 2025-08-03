import streamlit as st
import tempfile
import os
import shutil
import openai
import whisper
from moviepy import VideoFileClip
import ffmpeg
import re

import subprocess

try:
        version = subprocess.check_output(['ffmpeg', '-version']).decode()
        st.info(f"FFmpeg found:\n{version.splitlines()[0]}")
except Exception as e:
        st.error(f"FFmpeg not found: {e}")

# -- CONFIG --
import os
openai.api_key = os.environ.get("OPENAI_API_KEY")

 

st.title("AI Video Shortener (with Robust FFmpeg Debugging)")
st.markdown(
    """
    Upload a video, and this AI app will:
    - Transcribe it
    - Find the most interesting highlight (auto, with AI)
    - Create proper subtitles
    - Let you download your highlight video
    """
)

desired_length = st.slider("Length of summary video (seconds)", 30, 180, 60, step=10)
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
#subtitle_language = st.selectbox("Subtitle language", ["Original"])  # Ready for extension
subtitle_language = st.selectbox("Subtitle language", ["Original", "Arabic", "English"])

if uploaded_file:
    # --- Save uploaded video to temp file ---
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.success(f"File uploaded: {uploaded_file.name}")

    try:
        # --- Transcription with Whisper ---
        with st.spinner("Transcribing with Whisper..."):
            model = whisper.load_model("small")
            result = model.transcribe(video_path)
            transcript = result['text']
            segments = result['segments']
        st.text_area("Transcript", transcript, height=200)

        # --- Use LLM to pick best highlight window ---
        with st.spinner("Selecting best highlight window (AI)..."):
            segments_text = "\n".join([f"{s['start']:.1f}-{s['end']:.1f}: {s['text']}" for s in segments])
            prompt = (
                f"Given these transcript segments with timestamps, pick the best {desired_length}-second window. "
                f"Reply with only the start and end times in seconds, separated by a dash (e.g., 14.2-74.2):\n\n"
                f"{segments_text}"
            )
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                temperature=0.1,
            )
            gpt_reply = response.choices[0].message.content.strip()
            numbers = re.findall(r"\d+(?:\.\d+)?", gpt_reply)
            if len(numbers) >= 2:
                start_time = float(numbers[0])
                end_time = float(numbers[1])
            elif len(numbers) == 1:
                start_time = float(numbers[0])
                end_time = min(start_time + desired_length, segments[-1]['end'])
            else:
                start_time = 0
                end_time = min(desired_length, segments[-1]['end'])
            # Clamp to duration
            with VideoFileClip(video_path) as clip:
                video_duration = clip.duration
            if end_time > video_duration:
                end_time = video_duration
            if start_time > end_time:
                start_time = max(0, end_time - desired_length)
            if end_time - start_time > desired_length:
                end_time = start_time + desired_length
            st.info(f"Selected highlight window: {start_time:.2f} to {end_time:.2f} seconds (Video duration: {video_duration:.2f})")

        # --- Extract the highlight (with MoviePy) ---
        with st.spinner("Extracting highlight clip..."):
            try:
                highlight_clip = VideoFileClip(video_path).subclipped(start_time, end_time)
                # Copy to local file (not temp, for ffmpeg on Windows)
                highlight_local = "highlight_to_burn.mp4"
                highlight_clip.write_videofile(highlight_local, codec='libx264', audio_codec='aac')
            except Exception as e:
                st.error(f"FFmpeg error (highlight step): {e}")
                st.stop()

        # --- Build SRT subtitle file for the highlight window ---
        def format_time_srt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s_ = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02}:{m:02}:{s_:02},{ms:03}"

        def write_srt(segments, path, highlight_start, highlight_end):
            idx = 1
            with open(path, "w", encoding="utf-8") as f:
                for seg in segments:
                    seg_start = seg['start']
                    seg_end = seg['end']
                    # Only include segments in highlight window
                    if seg_end <= highlight_start or seg_start >= highlight_end:
                        continue
                    seg_start_clamped = max(seg_start, highlight_start) - highlight_start
                    seg_end_clamped = min(seg_end, highlight_end) - highlight_start
                    text = seg['text']
                    f.write(f"{idx}\n{format_time_srt(seg_start_clamped)} --> {format_time_srt(seg_end_clamped)}\n{text}\n\n")
                    idx += 1

        srt_local = "highlight_to_burn.srt"
        write_srt(segments, srt_local, start_time, end_time)


  def translate_segments(segments, target_language):
    if target_language == "Original":
        return segments
    translated = []
    for seg in segments:
        gpt_prompt = f"Translate this to {target_language}:\n\n{seg['text']}"
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": gpt_prompt}],
                max_tokens=128,
                temperature=0.3,
            )
            seg_copy = seg.copy()
            seg_copy['text'] = resp.choices[0].message.content.strip()
            translated.append(seg_copy)
        except Exception as e:
            seg_copy = seg.copy()
            seg_copy['text'] = seg['text']
            translated.append(seg_copy)
    return translated

segments_for_srt = translate_segments(segments, subtitle_language)
# Then use segments_for_srt for your SRT writing function

        # --- Preview the SRT file ---
        with open(srt_local, "r", encoding="utf-8") as srtf:
            srt_content = srtf.read()
        st.text_area("SRT file preview", srt_content, height=200)

        # --- Burn subtitles with ffmpeg, robust error capture ---
        final_path = "final_with_subs.mp4"
        with st.spinner("Adding subtitles with ffmpeg..."):
            try:
                ffmpeg.input(highlight_local).output(
                    final_path,
                    vf=f'subtitles={srt_local}'
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            except ffmpeg.Error as e:
                st.error(
                    f"FFmpeg error (subtitle step):\n\n"
                    f"{e.stderr.decode('utf-8', errors='ignore') if hasattr(e.stderr, 'decode') else e.stderr}"
                )
                st.stop()

        # --- Preview and download ---
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("Download Your Video Short", f, file_name="highlight_with_subs.mp4", mime="video/mp4")
        st.success("All done! 🎉")

    except Exception as e:
        st.error(f"An error occurred: {e}")
