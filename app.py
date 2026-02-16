# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: app_The_Hidden_Entomology_of_Coraline.py
# Bytecode version: 3.11a7e (3495)
# Source timestamp: 2025-12-25 16:06:42 UTC (1766678802)

import streamlit as st
import json
import os
import subprocess
import shutil
import time
import requests
import re
from pathlib import Path
import tempfile
import uuid
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip, TextClip
import whisper
from gtts import gTTS
from google import genai
from google.genai import types
import wave
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import glob
import base64
from PIL import Image
import io
import math
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
try:
    import cv2
except ImportError:
    cv2 = None
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- TRAILER & SCENE CACHE ---
TRAILER_PATH = None
TRAILER_DURATION = 0
TRAILER_YOUTUBE_URL = None
SCENE_CLIPS_DIR = None
SCENE_CATALOG = None

# Character images for animated explanations
# Character images for animated explanations
CHARACTER_IMAGES = {
    'beldam': 'character_assets/beldam.webp',
    'coraline': 'character_assets/coraline.webp',
    'wybie': 'character_assets/wybie.webp',
    'beldam_coraline': 'character_assets/beldam_coraline.webp'
}

AVATAR_IMAGES = {
    "neutral": "avatar_assets/neutral.png",
    "happy": "avatar_assets/happy.png",
    "smug": "avatar_assets/smug.png",
    "angry": "avatar_assets/angry.png",
    "confused": "avatar_assets/confused.png",
    "skeptical": "avatar_assets/skeptical.png",
    "surprised": "avatar_assets/surprised.png",
    "explaining": "avatar_assets/explaining.png"
}

# --- KOKORO CONFIGURATION ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
KOKORO_MODEL_PATH = os.path.join(APP_ROOT, "kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.path.join(APP_ROOT, "voices-v1.0.bin")
KOKORO_SUPPORTED_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam",
    "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa"]



def load_api_keys():
    try:
        with open('api_keys_config.json', 'r') as f:
            keys = json.load(f)
            return keys
    except:
        pass
    return {}

config_keys = load_api_keys()
DEFAULT_CLIP_MAPPING = {1: 'CORALINE FILME COMPLETO DUBLADO ( PARTE 3 )..mp4', 2: 'CORALINE FILME COMPLETO DUBLADO (PARTE 5).mp4', 3: 'CORALINE FILME COMPLETO DUBLADO ( PARTE 8 ).mp4', 4: 'CORALINE FILME COMPLETO DUBLADO ( PARTE 11).mp4', 5: 'CORALINE FILME COMPLETO DUBLADO (PARTE 12).mp4', 6: 'CORALINE FILME COMPLETO DUBLADO (PARTE 13).mp4', 7: 'CORALINE FILME COMPLETO DUBLADO (PARTE 14).mp4', 8: 'CORALINE FILME COMPLETO DUBLADO (PARTE 15).mp4', 9: 'CORALINE FILME COMPLETO DUBLADO (PARTE 16).mp4'}

def parse_all_clip_markdowns(base_dir='.'):
    """
    Parses all Clip*.md files to extract filename and description info.
    """
    all_clips = []
    md_files = sorted(glob.glob(os.path.join(base_dir, 'Clip*.md')))
    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Regex to find filenames and their blocks
            matches = re.finditer(r'## Filename: (.*?)\s*\n(.*?)(?=## Filename|\Z)', content, re.DOTALL)
            for m in matches:
                filename = m.group(1).strip()
                desc_block = m.group(2).strip()
                # Clean up description
                all_clips.append({'filename': filename, 'description': desc_block[:500]}) # Limit desc length
    return all_clips

def find_clip_path(filename):
    """
    Locates the actual video file for a given filename.
    Assumes structure: media/extracted_clips/Clip_XX/filename
    Checks both local 'media' and parent '../media'.
    """
    # Handle non-string inputs (AI sometimes returns numbers)
    if filename is None:
        return None
    if not isinstance(filename, str):
        print(f"[WARNING] find_clip_path received non-string filename: {filename} (type: {type(filename).__name__})")
        filename = str(filename)
    
    # Define search roots: current dir and parent dir
    search_roots = ['media', os.path.join('..', 'media'), '.', '..']
    
    # 1. Exact Match Strategy
    match = re.search(r'(Clip_\d+)', filename)
    if match:
        folder_name = match.group(1)
        for root in search_roots:
            candidate = os.path.join(root, 'extracted_clips', folder_name, filename)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    
    # 2. Recursive Search (Exact & Case-Insensitive)
    filename_lower = filename.lower()
    
    for search_root in search_roots:
        if os.path.exists(search_root):
            for root, dirs, files in os.walk(search_root):
                # Speed optimization: Skip hidden folders or build folders
                if 'build' in root or '__pycache__' in root:
                    continue
                    
                # Exact check
                if filename in files:
                    return os.path.abspath(os.path.join(root, filename))
                
                # Case-insensitive check
                for f in files:
                    if f.lower() == filename_lower:
                        return os.path.abspath(os.path.join(root, f))
                        
                # 3. Fuzzy Strategy: Check if filename is IN the file (or vice versa)
                # Useful if LLM truncates "Clip_01_Scene_001.mp4" to "Scene_001.mp4"
                for f in files:
                    if f.endswith('.mp4'):
                        if filename in f or (len(filename) > 10 and f in filename):
                             # Verify it's not a false positive
                             return os.path.abspath(os.path.join(root, f))
                             
    return None

def extract_clip_ffmpeg(video_path, start_time, end_time, output_path):
    """
    Cuts the video file from start_time to end_time using FFMPEG.
    Format for times: MM:SS or HH:MM:SS
    """
    if len(start_time) == 5:
        start_time = '00:' + start_time
    if len(end_time) == 5:
        end_time = '00:' + end_time
    cmd = ['ffmpeg', '-y', '-i', video_path, '-ss', start_time, '-to', end_time, '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', output_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if os.path.exists(output_path):
            return output_path
    except Exception as e:
        return None

def get_file_duration(path):
    """Returns duration in seconds using ffprobe"""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path]
        return float(subprocess.check_output(cmd).decode().strip())
    except Exception as e:
        print(f'Error getting duration for {path}: {e}')
        return 0

def render_animated_image(scene_id, image_configs, duration=5, bounce_type='elastic', base_dir='.'):
    """
    Creates an animated video with a white background and multiple images bouncing in at different positions.
    Uses PIL for FAST rendering - no Selenium overhead!
    Uses 8 threads for parallel frame generation.
    
    Args:
        scene_id: Unique identifier for this scene
        image_configs: List of dicts, each with:
            - 'image_path': path to the image
            - 'position': 'left', 'right', 'center', 'bottom_left', 'bottom_right', 'top_left', 'top_right'
        duration: Duration in seconds
        bounce_type: 'elastic', 'bounce', 'spring', 'slide_bounce'
        base_dir: Base directory for locating images
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image, ImageDraw, ImageFilter
    
    # Normalize to list format
    if isinstance(image_configs, str):
        image_configs = [{'image_path': image_configs, 'position': 'center'}]
    elif isinstance(image_configs, dict):
        image_configs = [image_configs]
    
    # Use absolute paths to prevent issues when CWD is different
    base_output_dir = os.path.dirname(os.path.abspath(__file__))
    media_dir = os.path.join(base_output_dir, 'media')
    os.makedirs(media_dir, exist_ok=True)
    output_file = os.path.join(media_dir, f'animated_img_{scene_id}.mp4')
    temp_dir = os.path.join(base_output_dir, 'temp_frames', f'anim_{scene_id}')
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"[DEBUG] Animated image output: {output_file}")
    print(f"[DEBUG] Temp frames dir: {temp_dir}")
    
    fps = 30
    total_frames = int(duration * fps)
    anim_frames = int(1.5 * fps)  # 1.5 second animation
    
    # Canvas size
    WIDTH, HEIGHT = 1280, 720
    
    # Position mappings - returns (x, y) center point for each position
    # Position mappings - returns (x, y) center point for each position
    POSITION_MAP = {
        'center': (WIDTH // 2, HEIGHT // 2),
        'left': (WIDTH // 4, HEIGHT // 2),
        'right': (3 * WIDTH // 4, HEIGHT // 2),
        
        # Quadrants (centered within each 1/4 of screen)
        # Adjusted to be less "edge-hugging" and safe from cut-off
        'top_left': (int(WIDTH // 3.5), int(HEIGHT // 3.5)),      # ~365, 205
        'top_right': (int(WIDTH - WIDTH // 3.5), int(HEIGHT // 3.5)),
        'bottom_left': (int(WIDTH // 3.5), int(HEIGHT - HEIGHT // 3.5)), # ~365, 515
        'bottom_right': (int(WIDTH - WIDTH // 3.5), int(HEIGHT - HEIGHT // 3.5)),
    }
    
    # Calculate image size based on number of images
    num_images = len(image_configs)
    if num_images == 1:
        max_w, max_h = 500, 550
    elif num_images == 2:
        max_w, max_h = 380, 450
    else:
        max_w, max_h = 300, 350 # Reduced height slightly to prevent bottom cut-off
    
    # Load and prepare all character images
    loaded_images = []
    for config in image_configs:
        image_path = config.get('image_path', '')
        position = config.get('position', 'center')
        
        print(f"[DEBUG] Attempting to load image: {image_path}")
        
        if not os.path.exists(image_path):
            alt_path = os.path.join(base_dir, os.path.basename(image_path))
            print(f"[DEBUG] Image not at original path, trying: {alt_path}")
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                # Also try relative to the script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                alt_path2 = os.path.join(script_dir, os.path.basename(image_path))
                print(f"[DEBUG] Trying script-relative path: {alt_path2}")
                if os.path.exists(alt_path2):
                    image_path = alt_path2
                else:
                    print(f"[ERROR] Image not found at any path: {image_path}")
                    continue
        
        try:
            char_img = Image.open(image_path).convert('RGBA')
            
            # --- Background Removal ---
            # SKIP if file is already cleaned (starts with "clean_")
            filename = os.path.basename(image_path)
            if not filename.startswith('clean_'):
                try:
                    from rembg import remove
                    print(f"[DEBUG] Removing background for {image_path} using rembg...")
                    char_img = remove(char_img)
                except ImportError:
                    print("[WARNING] 'rembg' not installed. Using simple black-removal fallback.")
                    # Fallback: Make near-black pixels transparent
                    datas = char_img.getdata()
                    new_data = []
                    for item in datas:
                        # If pixel is very dark (black/dark gray), make it transparent
                        if item[0] < 50 and item[1] < 50 and item[2] < 50:
                            new_data.append((255, 255, 255, 0))
                        else:
                            new_data.append(item)
                    char_img.putdata(new_data)
                except Exception as e:
                    print(f"[ERROR] Background removal failed: {e}")
            else:
                print(f"[DEBUG] Skipping background removal for pre-cleaned image: {filename}")
            # --------------------------

            # Scale to fit
            ratio = min(max_w / char_img.width, max_h / char_img.height)
            new_size = (int(char_img.width * ratio), int(char_img.height * ratio))
            char_img = char_img.resize(new_size, Image.Resampling.LANCZOS)
            loaded_images.append({
                'img': char_img,
                'position': position,
                'center': POSITION_MAP.get(position, POSITION_MAP['center'])
            })
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
    
    if not loaded_images:
        print("No images loaded successfully")
        return None
    
    # Easing functions for different bounce types
    def ease_out_elastic(t):
        if t == 0 or t == 1:
            return t
        p = 0.3
        s = p / 4
        return pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1
    
    def ease_out_bounce(t):
        if t < 1/2.75:
            return 7.5625 * t * t
        elif t < 2/2.75:
            t -= 1.5/2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5/2.75:
            t -= 2.25/2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625/2.75
            return 7.5625 * t * t + 0.984375
    
    def get_animation_state(frame_num, img_index=0):
        """Returns (scale, y_offset, opacity, rotation, x_offset) for a given frame"""
        # Stagger animation start for multiple images
        stagger_delay = img_index * 5  # 5 frames delay between each image
        adjusted_frame = max(0, frame_num - stagger_delay)
        
        if adjusted_frame >= anim_frames:
            return (1.0, 0, 1.0, 0, 0)  # Final resting state
        
        t = adjusted_frame / anim_frames  # 0 to 1
        
        if bounce_type == 'elastic':
            progress = ease_out_elastic(t)
            scale = progress
            y_offset = (1 - progress) * -200
            opacity = min(1.0, t * 3)
            rotation = 0
            x_offset = 0
        elif bounce_type == 'bounce':
            progress = ease_out_bounce(t)
            scale = 1.0
            y_offset = (1 - progress) * -400
            opacity = min(1.0, t * 3)
            rotation = (1 - progress) * -10
            x_offset = 0
        elif bounce_type == 'spring':
            progress = ease_out_elastic(t)
            scale = 0.3 + progress * 0.7
            y_offset = 0
            opacity = min(1.0, t * 3)
            rotation = (1 - progress) * -180
            x_offset = 0
        elif bounce_type == 'slide_bounce':
            progress = ease_out_elastic(t)
            scale = 0.7 + progress * 0.3
            y_offset = 0
            # Alternate slide direction for multiple images
            slide_dir = -1 if img_index % 2 == 0 else 1
            x_offset = (1 - progress) * slide_dir * 600
            opacity = min(1.0, t * 3)
            rotation = 0
        else:
            progress = ease_out_elastic(t)
            scale = progress
            y_offset = (1 - progress) * -200
            opacity = min(1.0, t * 3)
            rotation = 0
            x_offset = 0
        
        return (scale, y_offset, opacity, rotation, x_offset)
    
    def create_background():
        """Create a clean white background"""
        bg = Image.new('RGBA', (WIDTH, HEIGHT), (255, 255, 255, 255))
        return bg
    
    # Pre-create background (reused for all frames)
    background = create_background()
    
    def render_frame(frame_num):
        """Render a single frame and save it"""
        # Create frame from background
        frame = background.copy()
        
        # Render each image
        for img_idx, img_data in enumerate(loaded_images):
            char_img = img_data['img']
            center_x, center_y = img_data['center']
            
            state = get_animation_state(frame_num, img_idx)
            scale, y_offset, opacity, rotation, x_offset = state
            
            # Skip if not visible yet
            if scale < 0.01:
                continue
            
            scaled_size = (max(1, int(char_img.width * scale)), max(1, int(char_img.height * scale)))
            scaled_char = char_img.resize(scaled_size, Image.Resampling.LANCZOS)
            
            # Apply rotation if needed
            if abs(rotation) > 0.5:
                scaled_char = scaled_char.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
            
            # Apply opacity
            if opacity < 1.0:
                alpha = scaled_char.split()[3]
                alpha = alpha.point(lambda x: int(x * opacity))
                scaled_char.putalpha(alpha)
            
            # Calculate position based on center point + offsets
            pos_x = center_x - scaled_char.width // 2 + int(x_offset)
            pos_y = center_y - scaled_char.height // 2 + int(y_offset)
            
            # Paste character directly (no shadow)
            frame.paste(scaled_char, (pos_x, pos_y), scaled_char)
        
        # Convert to RGB and save
        frame_rgb = frame.convert('RGB')
        frame_path = f'{temp_dir}/frame_{frame_num:04d}.png'
        frame_rgb.save(frame_path, 'PNG', optimize=False)
        return frame_path
    
    try:
        print(f'🎨 Rendering {total_frames} frames with 8 threads ({len(loaded_images)} images)...')
        
        # Use 8 threads to render frames in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(render_frame, i): i for i in range(total_frames)}
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 30 == 0:
                    print(f'   Progress: {completed}/{total_frames} frames')
        
        print(f'✅ All frames rendered! Assembling video...')
        
        # Assemble video with ffmpeg (8 threads)
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', f'{temp_dir}/frame_%04d.png',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-threads', '8',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            output_file
        ]
        print(f"[DEBUG] Running FFMPEG: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[ERROR] FFMPEG failed: {result.stderr.decode()}")
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f'✅ Animation complete: {output_file} ({file_size} bytes)')
            if file_size < 1000:
                print(f"[WARNING] Output file is suspiciously small!")
            return output_file
        else:
            print(f"[ERROR] Output file not created: {output_file}")
        return None
    except Exception as e:
        import traceback
        print(f'Animated image render error: {e}')
        traceback.print_exc()
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


# ============================================
# NEW BETTER FEATURES FROM AUTO THEORY
# ============================================

def get_cache_paths(movie_name):
    """
    Returns all cache file paths for a given movie.
    Centralizes cache path management for consistency.
    """
    movie_slug = movie_name.lower().replace(' ', '_')
    build_dir = os.path.abspath('build')
    os.makedirs(build_dir, exist_ok=True)
    
    return {
        'trailer': os.path.join(build_dir, f'{movie_slug}_trailer.mp4'),
        'url_cache': os.path.join(build_dir, f'{movie_slug}_trailer_url.txt'),
        'vector_cuts': os.path.join(build_dir, f'{movie_slug}_trailer_vector_cuts.json'),
        'scene_catalog': os.path.join(build_dir, f'{movie_slug}_scenes.json'),
        'scene_summary': os.path.join(build_dir, f'{movie_slug}_scenes_summary.txt'),
        'scene_clips_dir': os.path.join(build_dir, f'{movie_slug}_trailer_scenes'),
        'thumbnails_dir': os.path.join(build_dir, 'thumbnails'),
    }

def check_cache_status(movie_name):
    """
    Checks what's already cached for a movie.
    """
    paths = get_cache_paths(movie_name)
    status = {
        'trailer': os.path.exists(paths['trailer']),
        'url_cache': os.path.exists(paths['url_cache']),
        'vector_cuts': os.path.exists(paths['vector_cuts']),
        'scene_catalog': os.path.exists(paths['scene_catalog']),
        'scene_clips': os.path.exists(paths['scene_clips_dir']) and 
                       len([f for f in os.listdir(paths['scene_clips_dir']) if f.endswith('.mp4')]) > 0
                       if os.path.exists(paths['scene_clips_dir']) else False,
    }
    return status

def select_best_video_with_ollama(search_query, video_list, context="trailer"):
    """
    Uses Ollama (qwen2.5:1.5b) to intelligently pick the best video
    from a list of YouTube search results.
    """
    if not video_list: return None
    if len(video_list) == 1: return video_list[0]
    
    options_text = "\n".join([
        f"{i+1}. Title: {v.get('title', 'Unknown')}, Channel: {v.get('channel', 'Unknown')}, Duration: {v.get('duration_string', 'Unknown')}"
        for i, v in enumerate(video_list)
    ])
    
    prompt = f"""You are helping select the best video clip for a visual story.
    Topic/Query: {search_query}
    Search Results:
    {options_text}
    RULES:
    - Pick the video that seems most relevant to the topic.
    - If for a trailer, pick the OFFICIAL trailer (not fan-made, not review, not reaction).
    - Prefer higher quality and official channels.
    Reply with ONLY the number (1, 2, 3, etc.) of the best choice. Just the number, nothing else."""

    try:
        payload = {
            'model': 'qwen2.5:1.5b',
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.1, 'num_ctx': 2048}
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        answer = result.get('response', '').strip()
        match = re.search(r'\d+', answer)
        if match:
            choice = int(match.group()) - 1
            if 0 <= choice < len(video_list):
                return video_list[choice]
    except:
        pass
    return video_list[0]

def get_best_segment_from_youtube_via_gemini(youtube_url, query_text, duration_needed, handler):
    """
    Sends YouTube URL directly to Gemini to find the best segment timestamps.
    """
    prompt = f"""You are analyzing this YouTube video: {youtube_url}
    I need a clip that matches this description: "{query_text}"
    The clip needs to be approximately {duration_needed} seconds long.
    
    Identify the BEST start time (in seconds) for such a clip.
    Return ONLY a single number representing the start time in seconds.
    Example: 45.5"""
    
    try:
        response = handler.generate_content(contents=prompt)
        text = response.text.strip()
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            start = float(match.group(1))
            return start, start + duration_needed
    except Exception as e:
        print(f"Gemini segment detection failed: {e}")
    
    return 0, duration_needed

def download_trailer(movie_name, force=False, manual_url=None, handler=None):
    """
    Downloads trailer and caches it.
    """
    global TRAILER_PATH, TRAILER_DURATION, TRAILER_YOUTUBE_URL
    paths = get_cache_paths(movie_name)
    
    if not force and os.path.exists(paths['trailer']):
        TRAILER_PATH = paths['trailer']
        TRAILER_DURATION = get_file_duration(TRAILER_PATH)
        if os.path.exists(paths['url_cache']):
            with open(paths['url_cache'], 'r') as f:
                TRAILER_YOUTUBE_URL = f.read().strip()
        return TRAILER_PATH
        
    video_url = manual_url
    if not video_url:
        search_query = f"ytsearch5:{movie_name} official trailer"
        cmd_search = ['yt-dlp', search_query, '--dump-json', '--no-download', '--flat-playlist']
        try:
            res = subprocess.run(cmd_search, capture_output=True, text=True, timeout=60)
            video_list = []
            for line in res.stdout.strip().split('\n'):
                if line:
                    v = json.loads(line)
                    video_list.append({
                        'id': v.get('id'),
                        'url': f"https://www.youtube.com/watch?v={v.get('id')}",
                        'title': v.get('title', 'Unknown'),
                        'duration': v.get('duration', 0),
                        'duration_string': v.get('duration_string', 'Unknown')
                    })
            if video_list:
                best = select_best_video_with_ollama(movie_name, video_list, context="trailer")
                video_url = best['url']
        except Exception as e:
            print(f"Trailer search failed: {e}")
            return None
            
    if not video_url: return None
    
    # Download
    cmd_dl = [
        'yt-dlp', video_url,
        '--output', paths['trailer'],
        '--format', 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        '--force-overwrites', '--quiet'
    ]
    try:
        subprocess.run(cmd_dl, check=True)
        if os.path.exists(paths['trailer']):
            TRAILER_PATH = paths['trailer']
            TRAILER_DURATION = get_file_duration(TRAILER_PATH)
            TRAILER_YOUTUBE_URL = video_url
            with open(paths['url_cache'], 'w') as f: f.write(video_url)
            return TRAILER_PATH
    except Exception as e:
        print(f"Download failed: {e}")
    return None

def detect_scenes_vector_embeddings(trailer_path, threshold=0.85, fps_process=12, min_scene_duration=1.0, save_clips=True):
    """
    CLIP-based scene detection using vector embeddings.
    """
    if SentenceTransformer is None or cv2 is None: return None
    
    trailer_hash = os.path.basename(trailer_path).replace('.mp4', '')
    paths = get_cache_paths(trailer_hash)
    
    if os.path.exists(paths['vector_cuts']):
        with open(paths['vector_cuts'], 'r') as f:
            data = json.load(f)
            return _create_scenes_from_cuts(data['cuts'], data['duration'], trailer_path, save_clips)
            
    model = SentenceTransformer('clip-ViT-B-32')
    cap = cv2.VideoCapture(trailer_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    step = max(1, int(fps / fps_process))
    
    prev_embedding = None
    cuts = [0.0]
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            embedding = model.encode(pil_image, convert_to_tensor=True)
            if prev_embedding is not None:
                score = util.cos_sim(embedding, prev_embedding).item()
                if score < threshold:
                    timestamp = frame_idx / fps
                    if timestamp - cuts[-1] >= min_scene_duration:
                        cuts.append(timestamp)
            prev_embedding = embedding
        frame_idx += 1
    cap.release()
    cuts.append(duration)
    
    cache_data = {'cuts': cuts, 'duration': duration, 'threshold': threshold}
    with open(paths['vector_cuts'], 'w') as f: json.dump(cache_data, f)
    
    return _create_scenes_from_cuts(cuts, duration, trailer_path, save_clips)

def _create_scenes_from_cuts(cuts, duration, trailer_path, save_clips=True):
    trailer_name = os.path.basename(trailer_path).replace('.mp4', '')
    build_dir = os.path.abspath('build')
    clips_dir = os.path.join(build_dir, f"{trailer_name}_scenes")
    os.makedirs(clips_dir, exist_ok=True)
    
    scenes = []
    for i in range(len(cuts) - 1):
        start, end = cuts[i], cuts[i+1]
        if end - start < 0.3: continue
        filename = f'scene_{len(scenes):04d}.mp4'
        clip_path = os.path.join(clips_dir, filename)
        scene = {'id': len(scenes), 'filename': filename, 'start': round(start, 2), 'end': round(end, 2), 'duration': round(end-start, 2), 'clip_path': clip_path}
        if save_clips and not os.path.exists(clip_path):
            ffmpeg_extract_subclip(trailer_path, start, end, targetname=clip_path)
        scenes.append(scene)
    return scenes

def deduplicate_scenes_visual_histogram(scenes, video_path, threshold=0.9):
    if not cv2 or not scenes: return scenes
    cap = cv2.VideoCapture(video_path)
    def get_hist(t):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret: return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    
    cleaned = []
    curr = scenes[0]
    for nxt in scenes[1:]:
        h_a, h_b = get_hist(curr['end'] - 0.1), get_hist(nxt['start'] + 0.1)
        if h_a is not None and h_b is not None and cv2.compareHist(h_a, h_b, cv2.HISTCMP_CORREL) > threshold:
            curr['end'] = nxt['end']
            curr['duration'] = curr['end'] - curr['start']
        else:
            cleaned.append(curr)
            curr = nxt
    cleaned.append(curr)
    cap.release()
    for i, s in enumerate(cleaned): s['id'] = i
    return cleaned

def describe_scenes_with_gemini_youtube(youtube_url, scenes, movie_name, handler):
    scene_info = "\n".join([f"- {s['filename']}: [{s['start']:.2f}s - {s['end']:.2f}s]" for s in scenes])
    prompt = f"Analyze the {movie_name} trailer at {youtube_url}. Describe each scene in this list: {scene_info}. Return JSON array with clip_id, filename, description."
    try:
        response = handler.client.models.generate_content(
            model='models/gemini-2.1-flash', 
            contents=types.Content(parts=[types.Part(file_data=types.FileData(file_uri=youtube_url)), types.Part(text=prompt)])
        )
        text = response.text
        if '```json' in text: text = text.split('```json')[1].split('```')[0]
        data = json.loads(text.strip())
        desc_map = {d.get('clip_id', d.get('id', -1)): d.get('description', '') for d in data}
        for s in scenes: s['description'] = desc_map.get(s['id'], f"Scene from {movie_name}")
    except:
        for s in scenes: s['description'] = f"Scene from {movie_name} [{s['start']:.1f}s]"
    return scenes

def catalog_trailer_scenes(trailer_path, movie_name, handler=None, force=False):
    paths = get_cache_paths(movie_name)
    if not force and os.path.exists(paths['scene_catalog']):
        with open(paths['scene_catalog'], 'r') as f: return json.load(f)
    
    scenes = detect_scenes_vector_embeddings(trailer_path)
    if not scenes: return []
    scenes = deduplicate_scenes_visual_histogram(scenes, trailer_path)
    
    if TRAILER_YOUTUBE_URL and handler:
        scenes = describe_scenes_with_gemini_youtube(TRAILER_YOUTUBE_URL, scenes, movie_name, handler)
    else:
        for s in scenes: s['description'] = f"Scene from {movie_name} [{s['start']:.1f}s]"
        
    with open(paths['scene_catalog'], 'w') as f: json.dump(scenes, f, indent=2)
    return scenes

def render_kenburns_image(scene_id, image_path, duration=5, effect='zoom_in'):
    if not os.path.exists(image_path): return None
    out = os.path.abspath(f'build/kb_{scene_id}.mp4')
    
    # Simple background cleaning logic
    try:
        from rembg import remove
        img = Image.open(image_path).convert('RGBA')
        img = remove(img)
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        bg.paste(img, (0, 0), img)
        proc_path = os.path.abspath(f"build/kb_tmp_{scene_id}.jpg")
        bg.convert('RGB').save(proc_path, quality=95)
        image_path = proc_path
    except: pass

    if effect == 'random': effect = random.choice(['zoom_in', 'zoom_out', 'pan_left', 'pan_right'])
    vf = f"scale=1664:936,zoompan=z='min(zoom+0.001,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={int(duration*30)}:s=1280x720:fps=30"
    if effect == 'zoom_out': vf = f"scale=1664:936,zoompan=z='if(eq(on,1),1.3,max(zoom-0.001,1))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={int(duration*30)}:s=1280x720:fps=30"
    
    cmd = ['ffmpeg', '-y', '-loop', '1', '-i', image_path, '-vf', vf, '-t', str(duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out if os.path.exists(out) else None

def render_avatar_explanation(scene_id, emotion, duration, background_clip=None):
    img_name = AVATAR_IMAGES.get(emotion, AVATAR_IMAGES.get('neutral'))
    if not img_name or not os.path.exists(img_name): return None
    out = os.path.abspath(f'build/av_{scene_id}.mp4')
    
    # Create blurred background
    bg_blur = os.path.abspath(f'build/av_bg_{scene_id}.mp4')
    source = background_clip if background_clip and os.path.exists(background_clip) else TRAILER_PATH
    if source:
        start = random.uniform(0, max(0, get_file_duration(source) - duration))
        cmd_blur = ['ffmpeg', '-y', '-ss', str(start), '-i', source, '-t', str(duration), '-vf', 'scale=1280:720,boxblur=25:25,eq=brightness=-0.1', '-an', bg_blur]
        subprocess.run(cmd_blur, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(bg_blur):
        cmd = ['ffmpeg', '-y', '-i', bg_blur, '-i', img_name, '-filter_complex', f"[1:v]scale=-1:792[av];[0:v][av]overlay=x='W/2-w/2':y=H-h+80:shortest=1", '-c:v', 'libx264', '-t', str(duration), out]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out
    return None

def generate_visual_plan_semantic_match(segments, scene_catalog, movie_name, handler=None):
    if not SentenceTransformer or not scene_catalog: return []
    model = SentenceTransformer('all-MiniLM-L6-v2')
    scene_embeddings = model.encode([s.get('description', '') for s in scene_catalog], convert_to_tensor=True)
    
    plan = []
    for seg in segments:
        emb = model.encode(seg['text'], convert_to_tensor=True)
        scores = util.cos_sim(emb, scene_embeddings)
        best_idx = np.argmax(scores.cpu().numpy())
        best_scene = scene_catalog[best_idx]
        score = scores[0][best_idx].item()
        
        item = {'text': seg['text'], 'visual_tool': 'MOVIE_CLIP' if score > 0.5 else 'KENBURNS_IMAGE', 
                'clip_num': best_scene['id'], 'start_time': best_scene['start'], 'end_time': best_scene['end'],
                'description': best_scene['description'], 'match_score': score}
        plan.append(item)
    return plan



class GeminiHandler:
    def __init__(self, key_string, start_index=0):
        self.keys = [k.strip() for k in key_string.split(',') if k.strip()]
        self.current_index = start_index % len(self.keys) if self.keys else 0
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.keys:
            try:
                self.client = genai.Client(api_key=self.keys[self.current_index])
            except Exception as e:
                print(f'Failed to init client with key {self.current_index}: {e}')

    def rotate_key(self):
        if len(self.keys) <= 1:
            return False
        self.current_index = (self.current_index + 1) % len(self.keys)
        
        # Sync with session state for persistence across re-runs
        if 'gemini_key_offset' in st.session_state:
            st.session_state['gemini_key_offset'] = self.current_index
            
        st.toast(f'🔄 Switched to Key #{self.current_index + 1}', icon='🔄')
        print(f'Rotating to key index {self.current_index}')
        self._init_client()
        return True

    def generate_content(self, model=None, contents=None, config=None):
        """
        Generate content using Gemini. 
        Note: model is optional here to match OllamaHandler signature.
        """
        # Resolve which one is the prompt
        if contents is None:
            # Called as generate_content(prompt_text)
            actual_contents = model
            actual_model = 'gemini-2.5-flash'
        else:
            # Called as generate_content(model='...', contents='...') or generate_content('...', '...')
            actual_contents = contents
            actual_model = model if model else 'gemini-2.5-flash'
             
        # Actual logic starts here
        retries = len(self.keys)
        for attempt in range(retries + 1):
            try:
                if not self.client:
                    raise Exception('No Gemini Client initialized')
                
                # Make the API call
                response = self.client.models.generate_content(model=actual_model, contents=actual_contents, config=config)
                
                # SUCCESS! Automatically rotate to the next key for the *next* request (Load Balancing)
                # This ensures we don't hammer one key if multiple exist
                if len(self.keys) > 1:
                    self.rotate_key()
                    
                return response
                
            except Exception as e:
                error_str = str(e)
                retry_triggers = ['429', 'RESOURCE_EXHAUSTED', 'Quota exceeded', '503', 'UNAVAILABLE', 'overloaded']
                if any(t in error_str for t in retry_triggers):
                    st.warning(f'⚠️ Key #{self.current_index + 1} Error ({error_str[:50]}...). Rotating...')
                    if not self.rotate_key():
                        st.error('❌ All Gemini keys exhausted!')
                        raise e
                else:
                    raise e
        raise Exception('Failed to generate content after rotating through all keys.')


# ============================================
# OLLAMA LOCAL LLM HANDLER
# ============================================
OLLAMA_URL = "http://localhost:11434/api/generate"

class OllamaResponse:
    """Mimics Gemini response structure for compatibility"""
    def __init__(self, text):
        self.text = text

class OllamaHandler:
    """Handles local Ollama model inference"""
    def __init__(self, model_name='gemma3:4b'):
        self.model_name = model_name
        self.connected = False
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama server is running"""
        try:
            r = requests.get('http://localhost:11434/api/tags', timeout=2)
            self.connected = r.status_code == 200
        except:
            self.connected = False
    
    def generate_content(self, model=None, contents=None, config=None):
        """
        Generate text using Ollama. 
        Matches GeminiHandler signature for drop-in replacement.
        Note: 'model' param is ignored, uses self.model_name
        """
        if not self.connected:
            raise Exception('Ollama server is not running. Start with: ollama serve')
        
        try:
            # Check if thinking mode should be disabled (for Qwen3, DeepSeek-R1)
            disable_thinking = st.session_state.get('ollama_disable_thinking', False)
            
            # Prepare prompt - handle positional/keyword contents
            prompt = contents if contents is not None else model
            
            if not prompt:
                raise Exception('No prompt provided for generation')

            if disable_thinking:
                # Qwen3 and similar models support /nothink suffix
                prompt = f"{prompt}\n\n/nothink"
            
            # Dynamic Context Calculation
            # Estimate tokens: ~2.5 chars per token
            est_tokens = int(len(prompt) / 2.5)
            
            # User Optimization: Dynamic Context Allocation
            # Instead of forcing 32k (which uses massive RAM even for small inputs),
            # we allocate exactly what we need + a generous buffer for the response.
            # This prevents 24GB systems from hitting disk swap.
            final_ctx = est_tokens + 4096

            payload = {
                'model': self.model_name,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'num_ctx': int(final_ctx),
                    'repeat_penalty': 1.1,
                    'num_thread': os.cpu_count(),
                    'num_gpu': 99
                }
            }
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=None)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '')
            
            # Strip thinking blocks if present (DeepSeek-R1, Qwen3 with think mode)
            # These appear as <think>...</think> in the output
            import re
            generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL)
            
            return OllamaResponse(generated_text.strip())
            
        except requests.exceptions.Timeout:
            raise Exception('Ollama request timed out. Model may be too slow.')
        except Exception as e:
            raise Exception(f'Ollama generation failed: {e}')


def generate_dynamic_plan(handler, topic, segments_json):
    """
    Sends the movie analysis + user topic to Gemini to generate a video plan JSON.
    """
    segments_text = json.dumps(segments_json, indent=2)
    system_prompt = f'''You are a dark storyteller uncovering hidden secrets in 'Coraline'.

I will provide you with a 'MOVIE DATABASE' which contains timestamps and details of specific clips from the movie.

YOUR TASK:
Create a JSON structure for a THEORY video about: "{topic}".

**WRITING STYLE - THIS IS CRITICAL:**
- You are NOT a film critic praising Laika's animation.
- You ARE a detective uncovering a dark, hidden truth.
- Write like you're revealing a terrifying secret, not reviewing a movie.
- NEVER say things like "Laika masterfully shows..." or "the animators cleverly..."
- INSTEAD say things like "The Beldam has been watching for years. Here's the proof..." or "This detail proves something horrifying..."

**TONE EXAMPLES:**
❌ BAD: "Laika's brilliant animation subtly foreshadows the terror to come."
✅ GOOD: "Look at this. The doll's eyes follow Coraline everywhere. It's not just a toy - it's a spy."

❌ BAD: "The filmmakers masterfully depicted the neglected home."
✅ GOOD: "Coraline's parents are checked out. They barely notice her. And THAT is exactly what the Beldam was waiting for."

RULES:
1. The JSON must follow this EXACT format:
{{
  "title": "Video Title Here",
  "scenes": [
    {{
      "id": 1,
      "visual_tool": "MOVIE_CLIP",
      "clip_num": 1,
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "description": "What this clip shows",
      "script": "THEORY-STYLE voiceover - speak like uncovering a dark secret"
    }},
    {{
      "id": 2,
      "visual_tool": "MANIM",
      "code_or_prompt": "from manim import *\\nclass Example(Scene):\\n    def construct(self):\\n        text = Text('Example')\\n        self.play(Write(text))",
      "description": "Animation description",
      "script": "THEORY-STYLE voiceover - 2-4 sentences"
    }},
    {{
      "id": 3,
      "visual_tool": "STOCK_VIDEO",
      "code_or_prompt": "search query for pexels",
      "description": "Stock video description",
      "script": "THEORY-STYLE voiceover - 2-4 sentences"
    }}
  ]
}}

2. When using "MOVIE_CLIP", you MUST:
   - Use "clip_num" from the database (integer 1-9)
   - Use EXACT start_time and end_time from the MOVIE DATABASE below
   - Do NOT invent timestamps - only use ones that exist in the database

3. Mix in "MANIM" or "STOCK_VIDEO" scenes to make it engaging and explanatory.

4. **CRITICAL - SCRIPT REQUIREMENTS:**
   - NEVER include clip names, timestamps, or technical references like [Clip_01_Scene...] in scripts
   - Scripts are SPOKEN voiceover - write naturally as if speaking to a friend
   - Build suspense like you're telling a scary story around a campfire
   - Use hooks like "But here's what nobody noticed..." or "Watch this part closely..."
   - Make the viewer feel like they're discovering something hidden

5. Generate 5-10 scenes total.

--- MOVIE DATABASE ---
{segments_text}

Return ONLY valid JSON, no markdown formatting.
'''
    try:
        with st.status("🕵️‍♀️ Beldam is watching... (AI Working)", expanded=True) as status:
            st.write("📄 Formatting Movie Database for context...")
            time.sleep(0.1)
            st.write("🧠 Sending 20k+ tokens to Ollama (Prompt Eval)...")
            response = handler.generate_content(contents=system_prompt)
            st.write("✅ Response received!")
            status.update(label="✅ Plan Generated!", state="complete", expanded=False)
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        return json.loads(response_text.strip())
    except Exception as e:
        st.error(f'Gemini API Error: {e}')

@st.cache_resource
def load_whisper():
    return whisper.load_model('tiny')

def generate_silence(duration, output_path):
    cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono', '-t', str(duration), output_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def prepare_fixed_duration_audio(input_path, duration, output_path):
    """
    Ensures the audio file is exactly `duration` seconds.
    - If shorter: Pads with silence at the end.
    - If longer: Trims to duration.
    """
    # 1. Get current duration
    curr_dur = get_file_duration(input_path)
    if curr_dur <= 0:
        # Fallback if invalid
        return generate_silence(duration, output_path)
        
    if abs(curr_dur - duration) < 0.1:
        # Close enough, just copy
        shutil.copy(input_path, output_path)
        return output_path
        
    if curr_dur > duration:
        # Trim
        cmd = ['ffmpeg', '-y', '-i', input_path, '-t', str(duration), '-c', 'copy', output_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # Pad
        pad_dur = duration - curr_dur
        # Create silence pad
        pad_file = output_path.replace('.wav', '_pad.wav')
        generate_silence(pad_dur, pad_file)
        
        # Concat
        list_file = output_path.replace('.wav', '_list.txt')
        with open(list_file, 'w') as f:
            f.write(f"file '{os.path.abspath(input_path).replace(os.sep, '/')}'\n")
            f.write(f"file '{os.path.abspath(pad_file).replace(os.sep, '/')}'\n")
            
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy', output_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Cleanup
        if os.path.exists(pad_file): os.remove(pad_file)
        if os.path.exists(list_file): os.remove(list_file)
        
    return output_path

def create_radio_edit(handler, script_text, audio_filename='full_narrative.wav'):
    """
    1. Parses script for "## Chapter" markers.
    2. Generates TTS for each section.
    3. Inserts 3s silence between sections for Title Cards.
    4. Stitches all audio into master track.
    5. Returns master audio path and timed segments (including Title placeholders).
    """
    # Process script to remove brackets/timestamps for TTS (e.g. [Clip_XX...])
    def clean_for_tts(text):
        # Remove [Clip...] or [Timestamp] patterns
        return re.sub(r'\[.*?\]', '', text).strip()
        
    # check for chapters
    chapters = re.split(r'(##\s+.*)', script_text)
    
    # If no chapters found or split failed to look like we expect, treat as single block
    if len(chapters) < 2:
        # Just one block
        clean_text = clean_for_tts(script_text)
        audio_path = generate_voiceover(handler, 'master_track', clean_text)
        model = load_whisper()
        result = model.transcribe(audio_path, word_timestamps=False)
        timed_segments = []
        for segment in result['segments']:
            timed_segments.append({'id': segment['id'], 'start': segment['start'], 'end': segment['end'], 'text': segment['text'].strip()})
        return (audio_path, timed_segments)
    
    # Process chapters
    # re.split with groups returns [text, delimiter, text, delimiter, ...]
    # The first element might be intro text (before first ##)
    blocks = []
    
    # Check if starts with text or delimiter
    current_title = "Intro"
    
    # Iterate through split results
    # chapters[0] is text before first header (Intro)
    if chapters[0].strip():
        blocks.append({'title': 'Intro', 'text': chapters[0].strip()})
        
    for i in range(1, len(chapters), 2):
        header = chapters[i].replace('##', '').strip()
        content = chapters[i+1].strip() if i+1 < len(chapters) else ""
        if content:
            # Clean content for TTS
            content = clean_for_tts(content)
            blocks.append({'title': header, 'text': content})

    final_segments = []
    audio_files = []
    current_offset = 0.0
    
    temp_dir = 'build/audio_chunks'
    os.makedirs(temp_dir, exist_ok=True)
    
    model = load_whisper()
    
    for i, block in enumerate(blocks):
        # 1. Add Title Gap (if not first, or if we want title at start too? Let's say yes for chapters)
        # Actually, let's put title BEFORE the text.
        # But if it's "Intro", maybe no title card?
        # User said: "first explain then pause in voice and then some effect for that title"
        # Wait, "first explain... then pause... then title... then normal script"
        # This implies: Intro -> Title -> Chapter 1 -> Title -> Chapter 2
        
        # TITLE CARDS AND SFX DISABLED BY USER REQUEST
        # if block['title'] != 'Intro':
        #     # Create Sound Effect MIXED into the start of the next audio
        #     sfx_source = os.path.abspath('soundeffect.wav')
        #     # Check if exists
        #     sfx_path = None
        #     if os.path.exists(sfx_source):
        #         sfx_path = sfx_source
            
        #     # Add Title Marker (0 duration)
        #     final_segments.append({
        #         'id': f'title_{i}',
        #         'start': current_offset,
        #         'end': current_offset, # 0 duration logic, visuals will expand this
        #         'text': f"Chapter: {block['title']}",
        #         'visual_tool': 'TITLE_CARD',
        #         'description': f"Chapter Title: {block['title']}"
        #     })
        #     # Do NOT increment current_offset here, as we are not adding time yet
            
        # 2. Generate Audio for Text
        raw_audio_path = generate_voiceover(handler, f"part_{i}_raw", block['text'])
        
        # If this block had a title, MIX the SFX into the start of raw_audio_path
        final_part_path = raw_audio_path
        # if block['title'] != 'Intro' and sfx_path:
        #     mixed_path = f"{temp_dir}/part_{i}_mixed.wav"
        #     # AMIX filter: mix sfx and speech. 
        #     # inputs=2. duration=first (match speech length, or longest? If speech is short, SFX might cut. 
        #     # If SFX is longer than speech, duration=first will cut SFX. duration=longest will pad speech with silence.
        #     # Let's use duration=longest to ensure full SFX is heard if it's long, 
        #     # but usually speech is paragraphs long.
        #     # dropout_transition=0.
        #     cmd_mix = [
        #         'ffmpeg', '-y',
        #         '-i', raw_audio_path,
        #         '-i', sfx_path,
        #         '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0',
        #         '-c:a', 'pcm_s16le',
        #         mixed_path
        #     ]
        #     subprocess.run(cmd_mix, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        #     if os.path.exists(mixed_path):
        #         final_part_path = mixed_path

        # 3. Transcribe this part to get relative timings
        # We transcribe the RAW audio usually for accuracy, but mixed is fine too.
        # Use final_part_path.
        result = model.transcribe(final_part_path, word_timestamps=True)
        part_duration = get_file_duration(final_part_path)
        
        # Add to master list with offset
        for seg in result['segments']:
            # Capture words with adjusted offsets
            seg_words = []
            if 'words' in seg:
                for w in seg['words']:
                    seg_words.append({
                        'word': w['word'],
                        'start': w['start'] + current_offset,
                        'end': w['end'] + current_offset
                    })
            
            final_segments.append({
                'id': f"txt_{i}_{seg['id']}",
                'start': seg['start'] + current_offset,
                'end': seg['end'] + current_offset,
                'text': seg['text'].strip(),
                'visual_tool': 'PREMADE_CLIP', # Default suggestion
                'words': seg_words
            })
            
        audio_files.append(final_part_path)
        current_offset += part_duration
        
    # Stitch Audio
    concat_list_path = f'{temp_dir}/concat_list.txt'
    with open(concat_list_path, 'w') as f:
        for audio in audio_files:
            f.write(f"file '{os.path.abspath(audio).replace(os.sep, '/')}'\n")
            
    final_audio = os.path.abspath(audio_filename)
    subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', final_audio], check=True)
    
    return (final_audio, final_segments)

def generate_visual_plan_from_audio(handler, timed_segments, clip_list):
    """
    Feeds the timestamped sentences to Gemini and asks it to select the best pre-made clip.
    """
    transcript_context = json.dumps(timed_segments, indent=2)
    # clip_list is now the list of dicts {filename, description}
    # We might need to chunk this if it's too large, but for now let's try sending it.
    # To save tokens, maybe just send filename and a short summary if descriptions are long.
    
    # Let's create a concise version for the prompt
    concise_clips = "\n".join([f"Filename: {c['filename']}\nDescription: {c['description'][:200]}..." for c in clip_list])
    
    system_prompt = f'''
    You are an expert video editor. I have a voiceover track.
    
    INPUT DATA:
    1. TRANSCRIPT (with timestamps):
    {transcript_context}
    
    2. AVAILABLE VISUAL CLIPS (Pre-made files):
    {concise_clips}
    
    3. AVAILABLE CHARACTER IMAGES (For explanatory segments with white background + bounce animation):
    - "beldam" (BeldamOtherMoterHuman2.webp) - The Beldam / Other Mother in human-like form
    - "coraline" (Coraline_Jones_with_one_of_her_outfits.webp) - Coraline Jones character image
    - "wybie" (Other_Wybie_waving.webp) - Other Wybie waving
    - "beldam_coraline" (TheBeldamCoralineBetterRes.webp) - Beldam with Coraline together
    
    YOUR TASK:
    For every segment of the script, assign the BEST, MOST RELEVANT visual from the list that matches the spoken words.
    
    CRITICAL: PRIORITIZE RELEVANCE. If the exact same clip is CRITICAL for the next segment, you may use it, but try to find a different angle or section if possible.
    
    OUTPUT FORMAT: Return a JSON with visual_tool, clip_filename (or character_images + positions), description, and avatar_pose.
    
    RULES:
    1. **CHECK FOR TITLE CARDS**: If a segment has "visual_tool": "TITLE_CARD" and "text": "Title Name", YOU MUST KEEP IT AS IS.
       - Output exact same visual_tool "TITLE_CARD" for that segment.
       - Do not change it to PREMADE_CLIP.
    
    2. **VIDEO PLAN INSTRUCTIONS**:
       - **STRATEGY**: Mix "PREMADE_CLIP" with "ANIMATED_IMAGE" for variety. Use "AVATAR_CHAT" *sparingly* (only for "I" statements).
       - **PREMADE_CLIP**: Default choice for narration/analysis.
       - **ANIMATED_IMAGE**: 
         - Use when introducing or discussing a SPECIFIC CHARACTER (Beldam, Coraline, Wybie).
         - Shows character image on white background with bounce-in animation.
         - Great for "character reveal" moments or when focusing on a character's traits/motivations.
         - Use "character_images" array with position: "center", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right".
         - Use "bounce_type": "elastic", "bounce", "spring", or "slide_bounce".
         - Example: When saying "The Beldam is a master manipulator...", use ANIMATED_IMAGE with "beldam" character.
       - **AVATAR_CHAT**: 
         - Use for personal opinions, theory introductions, asking questions to the audience, or reacting to shocking details ("I think...", "Can you believe this?", "This gave me chills").
         - **FREQUENCY**: Use freely! Don't be shy. If a segment is personal or impactful, show the Avatar.
         - **BACKGROUND**: You **MUST** provide a "clip_filename" for AVATAR_CHAT segments too. Find a relevant clip to play in the background (it will be blurred).
         - If no specific clip fits, reuse the *previous* clip's filename. Don't leave it empty.
       - **AVATAR POSE**: **ALWAYS REQUIRED**. "neutral", "happy", "smug", "angry", "confused", "skeptical".
       - **EVEN FOR PREMADE_CLIP**: You MUST specify "avatar_pose" for the PiP overlay.
         - Narrating a scary part? -> "skeptical" or "confused".
         - Explaining a fact? -> "neutral" or "explaining".
         - Making a joke? -> "happy" or "smug".
    
    3. **IMPORTANT - Use ANIMATED_IMAGE for variety!**:
       - Don't just use PREMADE_CLIP for everything. Sprinkle in ANIMATED_IMAGE when discussing characters.
       - Aim for at least 2-3 ANIMATED_IMAGE segments per video for visual interest.

    4. "clip_filename" MUST normally match one of the filenames provided in the list exactly.
    
    5. **AVATAR POSE RULE**:
       - Set "avatar_pose" to a fitting emotion for EVERY segment.
       - FOR "I" STATEMENTS ("I think", "I noticed"): YOU MUST USE "AVATAR_CHAT". Do not hide the avatar when the narrator is giving a personal opinion.
       - This makes the video feel like a real person talking.
    
    6. **DYNAMIC TEXT OVERLAY (NEW!)**:
       - If a segment contains a **powerful statement**, a **list**, or a **key revelation**, set "dynamic_text_overlay": true.
       - This will display the spoken words on screen in a large, "brick-style" animated caption effect (like viral Shorts).
       - Use this for EMPHASIS. Don't use it for every segment. Use it for the "hook" or the "climax".
    
    7. **CLEAN JSON FORMAT**:
       - ALWAYS include "avatar_pose" for ALL segments.
    
    8. **TITLE CARDS (IMPORTANT)**:
       - IGNORE TITLE CARDS for this test (we disabled them in audio).
       - If you see them, just skip or make them 3s AVATAR_CHAT segments saying the title.
    
    9. **VARIETY VS. RELEVANCE (BALANCED RULE)**:
       - **Primary Goal**: Show the MOST RELEVANT visual for the spoken words.
       - **Secondary Goal**: Avoid repetitive visuals.
       - **Process**: 
         1. First, find the absolute best clip for the segment.
         2. Check if it is the same as the previous segment.
         3. If it IS the same, check if there is a *second-best* clip that is still highly relevant.
         4. If a good alternative exists, use the alternative.
         5. If NO good alternative exists (e.g. we are talking about a specific unqiue moment), then it is ACCEPTABLE to repeat the clip. **relevance > variety**.
    

    JSON FORMAT:
    {{
      "visual_plan": [
        {{
          "segment_id": "txt_0_0",
          "start": 0.0,
          "end": 4.5,
          "text": "Coraline is not just a story...",
          "visual_tool": "PREMADE_CLIP",
          "clip_filename": "Clip_01_Scene_video-Scene-001.mp4",
          "description": "Coraline looking bored...",
          "dynamic_text_overlay": false,
          "avatar_pose": "neutral"
        }},
        {{
           "segment_id": "txt_0_99",
           "start": 10.0,
           "end": 13.0,
           "text": "Chapter: The Beldam",
           "visual_tool": "TITLE_CARD",
           "description": "Chapter Title: The Beldam",
           "avatar_pose": "explaining"
        }},
        {{
          "segment_id": "txt_1_1",
          "start": 13.0,
          "end": 20.0,
          "text": "The Beldam is actually based on a praying mantis...",
          "visual_tool": "ANIMATED_IMAGE",
          "character_images": [
            {{"image": "beldam", "position": "center"}}
          ],
          "bounce_type": "elastic",
          "description": "Beldam character reveal - discussing her insect nature",
          "avatar_pose": "smart",
          "dynamic_text_overlay": true
        }},
        {{
           "segment_id": "txt_1_2",
           "start": 20.0,
           "end": 25.0,
           "text": "I personally think this is the scariest part...",
           "visual_tool": "AVATAR_CHAT",
           "avatar_pose": "confused",
           "clip_filename": "Clip_05_Scene_video...mp4", 
           "description": "Host reacting to the scene (Clip 05 in background)"
        }}
      ]
    }}
    
    CRITICAL: Respect TITLE_CARD segments if they appear in input!
    '''
    try:
        response = handler.generate_content(contents=system_prompt)
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        st.error(f"Planning Error: {e}")
        return {"visual_plan": []}

def get_path_from_id(clip_id, movie_folder_path):
    if not clip_id:
        return None
    try:
        filename = DEFAULT_CLIP_MAPPING.get(int(clip_id))
        if filename:
            return os.path.join(movie_folder_path, filename)
    except:
        pass
    return None

def generate_srt(visual_plan, output_srt='subtitles.srt'):
    """
    Generates an SRT subtitle file from the visual plan.
    """
    def format_time(seconds):
        millis = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        return f'{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}'
    with open(output_srt, 'w', encoding='utf-8') as f:
        counter = 1
        for i, segment in enumerate(visual_plan['visual_plan']):
            # Skip subtitles for Title Cards
            if segment.get('visual_tool') == 'TITLE_CARD':
                continue
            
            start_str = format_time(segment['start'])
            end_str = format_time(segment['end'])
            text = segment['text']
            f.write(f'{counter}\n')
            f.write(f'{start_str} --> {end_str}\n')
            f.write(f'{text}\n\n')
            counter += 1
    return os.path.abspath(output_srt)

def render_avatar_explanation(scene_id, emotion, duration=5, base_dir='.', slide_in=True, slide_out=True, bg_video_path=None):
    """
    Renders an avatar explanation clip with slide-in/slide-out animation.
    If bg_video_path is provided, it overlays the avatar on that video (blurred/darkened).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image
    
    # 1. Select Image
    filename = AVATAR_IMAGES.get(emotion, AVATAR_IMAGES['neutral'])
    image_path = os.path.join(base_dir, filename)
    
    if not os.path.exists(image_path):
        # Allow fallback to checking current dir or downloads
        if os.path.exists(filename):
            image_path = filename
        else:
            print(f"Avatar image not found: {image_path}")
            # Fallback to simple colored frame? Or return None?
            return None
            
    # Setup Output
    base_output_dir = os.path.dirname(os.path.abspath(__file__))
    media_dir = os.path.join(base_output_dir, 'media')
    os.makedirs(media_dir, exist_ok=True)
    output_file = os.path.join(media_dir, f'avatar_{scene_id}.mp4')
    temp_dir = os.path.join(base_output_dir, 'temp_frames', f'av_{scene_id}')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # settings
    WIDTH, HEIGHT = 1920, 1080 # High res matching animatoravatar.py usually, or 1280x720?
    # App creates 1280x720 usually. Let's stick to 1280x720 for consistency with other clips.
    # But animatoravatar.py used 1920x1080.
    # Let's use 1280x720 to avoid resizing issues later.
    WIDTH, HEIGHT = 1280, 720
    FPS = 30
    total_frames = int(duration * FPS)
    
    # Load Character
    try:
        char_img = Image.open(image_path).convert('RGBA')
        # Scaling: animatoravatar uses 1.15 * screen height. 
        # For 720p: 720 * 1.15 = ~828 px height
        target_h = int(HEIGHT * 1.15)
        scale_factor = target_h / char_img.height
        new_w = int(char_img.width * scale_factor)
        new_h = int(char_img.height * scale_factor)
        char_img = char_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error loading avatar: {e}")
        return None
        
    # Animation Parameters
    # Slide In: Right to Left
    # Final Position (Rest): Center X = 75% of screen width (Right aligned)
    # Start Position: Offscreen Right (Center X = WIDTH + Width/2 ???)
    
    final_center_x = int(WIDTH * 0.5)
    start_center_x = WIDTH + new_w 
    
    # Vertical: Hide bottom cutout
    offset_below_screen = 100 # Adjusted for 720p
    base_y = (HEIGHT - new_h) + offset_below_screen
    
    # Timeline
    slide_duration = 0.5 # Quicker (0.5s)
    stay_duration = duration - (2 * slide_duration) # Slide in, stay, slide out
    if stay_duration < 0: stay_duration = 0 # Short clip handling
    
    slide_frames = int(slide_duration * FPS)
    
    # Background
    # If using video background, make this transparent
    if bg_video_path:
        bg_color = (0, 0, 0, 0) # Transparent
        background = Image.new('RGBA', (WIDTH, HEIGHT), bg_color)
    else:
        bg_color = (30, 30, 30) # Dark Grey like animatoravatar
        background = Image.new('RGB', (WIDTH, HEIGHT), bg_color)
    
    def ease_out_expo(x):
        return 1 if x == 1 else 1 - math.pow(2, -10 * x)

    def render_av_frame(frame_num):
        t_sec = frame_num / FPS
        
        # Calculate X Position
        current_center_x = final_center_x
        
        # 1. Slide In (Snappy)
        if slide_in and t_sec < slide_duration:
            progress = t_sec / slide_duration
            eased = ease_out_expo(progress)
            current_center_x = start_center_x + (final_center_x - start_center_x) * eased
            
        # 2. Slide Out (Snappy)
        elif slide_out and t_sec > (duration - slide_duration):
            remaining = duration - t_sec
            progress_out = 1 - (remaining / slide_duration)
            eased_out = ease_in_back(progress_out)
            current_center_x = final_center_x + (start_center_x - final_center_x) * eased_out
            
        # Breathing / Alive Effect
        # A continuous sine wave for Y offset (Bobbing)
        # A slightly offset sine wave for Scale (Squash/Stretch)
        # A slow sine wave for Rotation
        
        # Bobbing
        bob_amount = 8.0 # pixels
        bob_speed = 3.0
        float_y = math.sin(t_sec * bob_speed) * bob_amount
        
        # Breathing (Squash/Stretch) - barely perceptible but adds life
        # Scale Y goes up when Bob goes down (anticipation)
        breath_speed = 3.0
        breath_amount = 0.02 # 2% stretch
        scale_y_mult = 1.0 + (math.sin(t_sec * breath_speed + math.pi) * breath_amount)
        scale_x_mult = 1.0 - (math.sin(t_sec * breath_speed + math.pi) * (breath_amount * 0.5)) # Preserve area roughly
        
        # Rotation (Idle sway)
        rot_amount = 1.5 # degrees
        rot_speed = 1.5
        rotation = math.sin(t_sec * rot_speed) * rot_amount
        
        # Apply Slide transition to rotation too (tilt forward when sliding in)
        if slide_in and t_sec < slide_duration:
            # Tilt forward (negative rot) then settle
            tilt = (1.0 - ease_out_expo(t_sec/slide_duration)) * -5.0
            rotation += tilt
            
        # Compose
        frame = background.copy()
        
        # Apply Transforms to Character
        # 1. Rotate
        # 2. Scale
        
        # Working Image
        work_img = char_img.copy()
        
        # Rotate
        if abs(rotation) > 0.1:
            work_img = work_img.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
            
        # Scale (Squash/Stretch)
        cur_w, cur_h = work_img.size
        final_w = int(cur_w * scale_x_mult)
        final_h = int(cur_h * scale_y_mult)
        if final_w > 0 and final_h > 0:
            work_img = work_img.resize((final_w, final_h), Image.Resampling.LANCZOS)
        
        # Calculate centering based on new size
        paste_x = int(current_center_x - (work_img.width // 2))
        paste_y = int(base_y + float_y - (work_img.height - new_h)) # Adjust for growth from bottom

        # Add shadow?
        # Simple shadow: black ellipse at bottom, scaled by height
        
        # Paste
        try:
            frame.paste(work_img, (paste_x, paste_y), work_img)
        except:
            frame.paste(work_img, (paste_x, paste_y))
            
        # Save
        # Save
        # Only convert to RGB if we DON'T have a background video (to keep transparency for overlay)
        if not bg_video_path and frame.mode != 'RGB':
            frame = frame.convert('RGB')
        # If we DO have a bg_video, we WANT RGBA (transparent background) for the overlay filter
            
        out_p = f'{temp_dir}/f_{frame_num:04d}.png'
        frame.save(out_p, optimize=False)
        return out_p

    # Parallel Render
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(render_av_frame, i) for i in range(total_frames)]
            for f in as_completed(futures):
                pass # Just wait
                
        # ffmpeg assemble
        # ffmpeg assemble - TWO STEP PROCESS for Smoothness
        
        # Step 1: Prepare Background Video
        bg_processed_path = os.path.join(temp_dir, 'bg_processed.mp4')
        
        if bg_video_path and os.path.exists(bg_video_path):
             # Determine Stretch Factor
             bg_dur = get_file_duration(bg_video_path)
             stretch_filter = ""
             if bg_dur > 0 and bg_dur < duration:
                 stretch_factor = duration / bg_dur
                 # Use a generous stretch limit
                 if stretch_factor < 10.0:
                     stretch_filter = f"setpts={stretch_factor}*PTS,"
                     st.write(f"      Processing BG: Stretch {stretch_factor:.2f}x")
            
             # Render processed background ONLY
             cmd_bg = [
                'ffmpeg', '-y',
                '-i', bg_video_path,
                '-vf', f'{stretch_filter}scale={int(WIDTH*0.95)}:{int(HEIGHT*0.95)}:force_original_aspect_ratio=decrease,pad=iw+10:ih+10:(ow-iw)/2:(oh-ih)/2:color=white,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,boxblur=lp=2:lr=2,eq=brightness=-0.05',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-t', str(duration),
                '-r', str(FPS),
                '-an', # Remove audio from BG
                bg_processed_path
             ]
             res_bg = subprocess.run(cmd_bg, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             if res_bg.returncode != 0:
                 st.error(f"BG Processing Failed: {res_bg.stderr.decode()}")
                 # Fallback to color will happen if file not created
        
        # Step 2: Overlay Avatar on Background
        if os.path.exists(bg_processed_path):
            # Use the processed video
            cmd_final = [
                'ffmpeg', '-y',
                '-i', bg_processed_path,
                '-framerate', str(FPS),
                '-i', f'{temp_dir}/f_%04d.png',
                '-filter_complex', '[0:v][1:v]overlay=0:0:format=auto',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-r', str(FPS),
                output_file
            ]
        else:
            # Fallback: Solid Color Background + Avatar
            cmd_final = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', f'color=c=black:s={WIDTH}x{HEIGHT}:r={FPS}:d={duration}',
                '-framerate', str(FPS),
                '-i', f'{temp_dir}/f_%04d.png',
                '-filter_complex', '[0:v][1:v]overlay=0:0:format=auto',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-r', str(FPS),
                output_file
            ]
            
        st.write("      Combining Avatar + Background...")
        res = subprocess.run(cmd_final, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
             print(f"Avatar FFMPEG Error: {res.stderr.decode()}")
             shutil.rmtree(temp_dir, ignore_errors=True)
             return None
        
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
             print(f"Avatar Output Missing/Empty: {output_file}")
             shutil.rmtree(temp_dir, ignore_errors=True)
             return None

        shutil.rmtree(temp_dir, ignore_errors=True)
        return output_file
        
    except Exception as e:
        print(f"Avatar Render Failed: {e}")
        return None

def generate_word_level_subtitles(output_path, words, start_time, end_time):
    """
    Generates an .ass subtitle file for the given words with 'brick' style appearance.
    """
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,70,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,150,1
"""
    # Style: Arial Black, Size 70, White, Black Outline (thickness 3), Bottom Center (Align 2), MarginV 150 (Lower Third)
    
    events = "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    
    def format_time(seconds):
        # m:ss.cs format for ASS
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        cs = int((s - int(s)) * 100)
        return f"{int(h)}:{int(m):02d}:{int(s):02d}.{cs:02d}"

    # Chunking logic
    chunk_size = 2 # words per chunk
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i+chunk_size]
        if not chunk: continue
        
        t_start = chunk[0]['start']
        # End time: start of next chunk or end of last word + 0.3s
        if i + chunk_size < len(words):
            t_end = words[i+chunk_size]['start']
        else:
            t_end = chunk[-1]['end'] + 0.3

        # Convert to relative time within the segment
        sub_start = t_start - start_time
        sub_end = t_end - start_time
        
        if sub_end <= 0: continue
        if sub_start < 0: sub_start = 0
        # If subtitles extend beyond the clip duration (end_time - start_time), cap them?
        # clip duration = end_time (abs) - start_time (abs)
        clip_dur = end_time - start_time
        if sub_end > clip_dur: sub_end = clip_dur
        if sub_start >= sub_end: continue

        text = " ".join([w['word'].strip() for w in chunk]).upper()
        
        events += f"Dialogue: 0,{format_time(sub_start)},{format_time(sub_end)},Default,,0,0,0,,{text}\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header + events)
    return output_path

def apply_brick_text_overlays(handler, video_path, audio_path, original_script, output_path='final_with_text.mp4'):
    """
    Post-render step: Applies "brick-style" text overlays to the final video.
    1. Runs Whisper on the final audio to get accurate word-level timestamps.
    2. Sends transcript + script to Gemini to select key phrases for overlay.
    3. Generates .ass subtitle file and burns it into the video.
    """
    import streamlit as st
    
    st.info('🔊 Analyzing final audio with Whisper (word-level)...')
    model = load_whisper()
    result = model.transcribe(audio_path, word_timestamps=True)
    
    # Build full word list with absolute timestamps
    all_words = []
    for seg in result['segments']:
        if 'words' in seg:
            for w in seg['words']:
                all_words.append({
                    'word': w['word'],
                    'start': w['start'],
                    'end': w['end']
                })
    
    if not all_words:
        st.error('No words detected in audio. Cannot apply overlays.')
        return None
    
    # Build transcript text
    full_transcript = result.get('text', '')
    
    st.info(f'📝 Found {len(all_words)} words.')
    
    # Check if manual phrases were provided
    if st.session_state.get('manual_text_overlay_phrases'):
        selected_phrases = st.session_state['manual_text_overlay_phrases']
        st.success(f'✅ Using {len(selected_phrases)} manually provided phrases (skipping API call).')
        # Clear after use to allow re-running with API if desired
        # st.session_state['manual_text_overlay_phrases'] = None  # Uncomment to clear after use
    else:
        st.info('Asking Gemini to select key phrases...')
        
        # Ask Gemini to select phrases
        prompt = f'''
        I have a video essay script and its word-level transcript with timestamps.
        
        ORIGINAL SCRIPT (what I intended to say):
        {original_script[:3000]}
        
        ACTUAL TRANSCRIPT (what Whisper heard, with timing):
        {full_transcript[:3000]}
        
        YOUR TASK:
        Select 5-10 KEY PHRASES (2-4 words each) that should appear as large, bold "brick-style" text overlays.
        These should be:
        - Impactful statements ("The door is fake", "She never escaped")
        - Key revelations ("The Beldam knew")
        - Hook phrases that grab attention
        
        Return a JSON list of objects:
        [
            {{
                "phrase": "the door is fake",
                "emphasis": "high"  // "high", "medium", or "low"
            }}
        ]
        
        RULES:
        - Use the EXACT words as they appear in the transcript (matching Whisper output).
        - Don't over-use this effect. Pick only the most powerful moments.
        - Return ONLY valid JSON, no markdown.
        '''
        
        try:
            resp = handler.generate_content(contents=prompt)
            text = resp.text.strip()
            if text.startswith('```json'): text = text[7:]
            if text.startswith('```'): text = text[3:]
            if text.endswith('```'): text = text[:-3]
            selected_phrases = json.loads(text.strip())
        except Exception as e:
            st.error(f'Gemini phrase selection failed: {e}')
            st.info('💡 Tip: You can bypass this by using the "Step 5 Bypass" section to manually provide phrases.')
            return None
    
    st.info(f'✅ {len(selected_phrases)} phrases ready. Generating subtitles...')
    
    # Match phrases to word timestamps
    # Build .ass file
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Brick,Arial Black,80,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,10,10,100,1
"""
    events = "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    
    def format_ass_time(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        cs = int((s - int(s)) * 100)
        return f"{int(h)}:{int(m):02d}:{int(s):02d}.{cs:02d}"
    
    matched_count = 0
    for phrase_obj in selected_phrases:
        phrase = phrase_obj.get('phrase', '').lower().strip()
        phrase_words = phrase.split()
        if not phrase_words:
            continue
        
        # Find the phrase in all_words
        for i in range(len(all_words) - len(phrase_words) + 1):
            # Check if words match
            match = True
            for j, pw in enumerate(phrase_words):
                if pw not in all_words[i+j]['word'].lower():
                    match = False
                    break
            
            if match:
                # Found the phrase
                t_start = all_words[i]['start']
                t_end = all_words[i + len(phrase_words) - 1]['end'] + 0.3  # Extend slightly
                display_text = ' '.join([all_words[i+k]['word'].strip() for k in range(len(phrase_words))]).upper()
                events += f"Dialogue: 0,{format_ass_time(t_start)},{format_ass_time(t_end)},Brick,,0,0,0,,{display_text}\n"
                matched_count += 1
                break  # Only match first occurrence
    
    if matched_count == 0:
        st.warning('No phrases matched in the transcript. Skipping overlay.')
        return video_path
    
    st.info(f'🧱 Matched {matched_count} phrases. Burning into video...')
    
    # Use generic filename
    ass_filename = 'overlays.ass' 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ass_path = os.path.join(script_dir, ass_filename)
    
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(header + events)
    
    output_abs = os.path.abspath(output_path)
    video_abs = os.path.abspath(video_path)
    
    # Use simple filename and run in the directory
    # This avoids all Windows drive letter/escaping madness
    cmd = [
        'ffmpeg', '-y',
        '-i', video_abs,
        '-vf', f"ass={ass_filename}",
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-c:a', 'copy',
        output_abs
    ]
    
    st.write(f"Running FFmpeg (safemode)...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=script_dir)
    
    if result.returncode != 0:
        stderr_text = result.stderr.decode('utf-8', errors='replace')
        st.error(f'FFmpeg overlay failed!')
        st.code(stderr_text[:1000])
        return None
    
    # Verify output
    out_dur = get_file_duration(output_abs)
    if out_dur < 1:
        st.error(f'Output video is too short ({out_dur}s). FFmpeg may have failed silently.')
        return None
    
    st.success(f'✅ Text overlays applied! Output: {output_path} ({out_dur:.1f}s)')
    return output_path

def get_circular_avatar_path(emotion, base_dir='.'):
    """
    Creates a temporary circular avatar image for PiP.
    """
    from PIL import Image, ImageOps, ImageDraw
    
    image_filename = AVATAR_IMAGES.get(emotion, AVATAR_IMAGES['neutral'])
    image_path = os.path.join(base_dir, image_filename)
    
    # Try finding the image in various locations
    possible_paths = [
        image_path,
        os.path.join(base_dir, 'avatar_assets', f'{emotion}.png'),
        image_filename
    ]
    
    found_path = None
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    if not found_path:
         return None
             
    try:
        # Create temp path
        temp_dir = 'build/temp_avatars'
        os.makedirs(temp_dir, exist_ok=True)
        out_path = os.path.join(temp_dir, f'pip_{emotion}.png')
        
        # Force re-creation to ensure update
        # if os.path.exists(out_path): return out_path
            
        img = Image.open(found_path).convert("RGBA")
        
        # Resize to reasonable size for PiP (e.g. 250x250)
        size = (250, 250)
        img = ImageOps.fit(img, size, centering=(0.5, 0.5))
        
        # 1. Create a solid white square base
        # This ensures there is NO transparency behind the avatar
        combined = Image.new("RGBA", size, (255, 255, 255, 255))
        
        # 2. Paste Avatar on top
        combined.paste(img, (0, 0), img)
        
        # 3. Create Circle Mask (L mode)
        # Everything white (255) will show the combined image
        # Everything black (0) will be transparent
        mask = Image.new('L', size, 0)
        draw_mask = ImageDraw.Draw(mask) 
        draw_mask.ellipse((10, 10, 240, 240), fill=255)
        
        # 4. Apply Mask to create Final Round Image
        final = Image.new('RGBA', size, (0,0,0,0))
        final.paste(combined, (0,0), mask)
        
        # 5. Add White Border on transparent layer
        draw_border = ImageDraw.Draw(final)
        draw_border.ellipse((10, 10, 240, 240), outline="white", width=5)
        
        final.save(out_path)
        return out_path
    except Exception as e:
        print(f"Error creating circle avatar: {e}")
        return None

# --- Wav2Lip Configuration ---
WAV2LIP_DIR = r"C:\Users\USER\Downloads\GovindsaVoid\Wav2Lip"
WAV2LIP_INFERENCE = os.path.join(WAV2LIP_DIR, "inference.py")
WAV2LIP_CHECKPOINT = os.path.join(WAV2LIP_DIR, "checkpoints", "Wav2Lip-SD-GAN.pt")

def create_circle_mask_assets(size=(250, 250)):
    """Generates mask and border assets for video circular crop."""
    try:
        from PIL import Image, ImageDraw
        assets_dir = 'build/assets_mask'
        os.makedirs(assets_dir, exist_ok=True)
        mask_path = os.path.join(assets_dir, 'circle_mask.png')
        border_path = os.path.join(assets_dir, 'circle_border.png')
        
        if os.path.exists(mask_path) and os.path.exists(border_path):
            return mask_path, border_path
            
        # 1. Mask (White Circle on Black) - For alphamerge
        # White = Keep (Opacity 1), Black = Remove (Opacity 0)
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((5, 5, size[0]-5, size[1]-5), fill=255) # Slight padding
        mask.save(mask_path)
        
        # 2. Border Overlay (Transparent with White Ring)
        border = Image.new('RGBA', size, (0,0,0,0))
        draw_b = ImageDraw.Draw(border)
        # Stroke width 5
        draw_b.ellipse((5, 5, size[0]-5, size[1]-5), outline="white", width=5)
        border.save(border_path)
        
        return mask_path, border_path
    except Exception as e:
        print(f"Error creating masks: {e}")
        return None, None

def generate_wav2lip_clip(face_path, audio_path, output_path):
    """
    Runs Wav2Lip inference to lip-sync the face (image/video) with audio.
    """
    # Verify paths
    if not os.path.exists(WAV2LIP_INFERENCE):
        st.error(f"Wav2Lip inference.py not found at: {WAV2LIP_INFERENCE}")
        return None
        
    # Ensure Absolute Paths (Critical because we change CWD)
    face_path = os.path.abspath(face_path)
    audio_path = os.path.abspath(audio_path)
    output_path = os.path.abspath(output_path)
        
    cmd = [
        "python", WAV2LIP_INFERENCE,
        "--checkpoint_path", WAV2LIP_CHECKPOINT,
        "--face", face_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--super_res",
        "--sharpen",
        "--resize_factor", "1" # Keep quality
    ]
    
    # We need to run this command with CWD as Wav2Lip dir because it imports local modules
    try:
        st.write(f"      👄 Generating LipSync: {os.path.basename(face_path)} + {os.path.basename(audio_path)}")
        # Using shell=True might be needed for python environment resolution? 
        # But subprocess.run with cwd should work if 'python' is in path.
        res = subprocess.run(cmd, cwd=WAV2LIP_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if res.returncode != 0:
            print(f"Wav2Lip Failed: {res.stderr.decode()}")
            st.error(f"Wav2Lip Error: {res.stderr.decode()[:200]}...")
            return None
            
        if os.path.exists(output_path):
            return output_path
    except Exception as e:
        st.error(f"Exeption running Wav2Lip: {e}")
    return None

def assemble_video_from_plan(audio_path, visual_plan, movie_folder_path, transcript_segments=None, output_filename='final_theory.mp4'):
    """
    Assembles the video using robust FFMPEG command line calls (bypassing MoviePy errors).
    """
    st.info('🚀 Starting Robust FFMPEG Assembly...')
    temp_dir = 'build/temp_clips'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    concat_list_path = os.path.abspath('build/concat_list.txt')
    srt_path = generate_srt(visual_plan, 'build/subtitles.srt')
    clip_files = []
    progress_bar = st.progress(0)
    total_segments = len(visual_plan['visual_plan'])
    
    # Track the last valid clip to use as fallback background
    last_valid_clip_path = None

    # Sync visuals to audio ground truth
    ground_truth_map = {s['id']: s for s in transcript_segments} if transcript_segments else {}
    
    for i, segment in enumerate(visual_plan['visual_plan']):
        # Force strict timing if ID matches
        if segment.get('segment_id') in ground_truth_map:
             gt = ground_truth_map[segment['segment_id']]
             segment['start'] = gt['start']
             segment['end'] = gt['end']
             
        duration = segment['end'] - segment['start']
        visual_tool = segment['visual_tool']
        clip_id = segment.get('clip_id_from_db')
        segment_filename = f'segment_{i:03d}.mp4'
        segment_path = os.path.join(temp_dir, segment_filename)
        segment_path_abs = os.path.abspath(segment_path)
        source_path = None
        seek_time = segment.get('seek_time', '00:00')
        if visual_tool == 'MOVIE_CLIP':
            source_path = get_path_from_id(clip_id, movie_folder_path)
        elif visual_tool == 'PREMADE_CLIP':
            filename = segment.get('clip_filename')
            if filename:
                source_path = find_clip_path(filename)
                if not source_path:
                    st.error(f"❌ Clip NOT Found: {filename} (Using placeholder)")
                else:
                    last_valid_clip_path = source_path
                seek_time = "00:00" # Always start from beginning of premade clip
        elif visual_tool == 'TITLE_CARD':
            # Render title card
            title_text = segment.get('text', 'Chapter')
            st.write(f'   🎨 Rendering Title Card: "{title_text}"...')
            source_path = render_title_card(f"seg_{i}", title_text, duration=duration)
            seek_time = "00:00"
        elif visual_tool == 'ANIMATED_IMAGE':
            # Render animated image with bounce effect
            # Support both new (character_images array) and old (character_image single) formats
            bounce_type = segment.get('bounce_type', 'elastic')
            
            # Build image configs list
            image_configs = []
            
            # New format: character_images is an array with image+position
            if 'character_images' in segment and isinstance(segment['character_images'], list):
                for img_config in segment['character_images']:
                    char_key = img_config.get('image', 'coraline')
                    position = img_config.get('position', 'center')
                    image_filename = CHARACTER_IMAGES.get(char_key)
                    if image_filename:
                        image_configs.append({
                            'image_path': os.path.join(movie_folder_path, image_filename),
                            'position': position
                        })
            # Old format: character_image is a single string
            elif 'character_image' in segment:
                char_key = segment.get('character_image', 'coraline')
                image_filename = CHARACTER_IMAGES.get(char_key)
                if image_filename:
                    image_configs.append({
                        'image_path': os.path.join(movie_folder_path, image_filename),
                        'position': 'center'
                    })
            
            if image_configs:
                st.write(f'   🎨 Rendering animated images ({len(image_configs)} chars, {bounce_type} bounce)...')
                rendered_clip = render_animated_image(
                    scene_id=f"seg_{i}",
                    image_configs=image_configs,
                    duration=duration,
                    bounce_type=bounce_type,
                    base_dir=movie_folder_path
                )
                if rendered_clip and os.path.exists(rendered_clip):
                    source_path = rendered_clip
                    seek_time = "00:00"
        elif visual_tool == 'MANIM':
            manim_path = f"media/manim_{segment['segment_id']}.mp4"
            if os.path.exists(manim_path):
                source_path = manim_path
        elif visual_tool == 'AVATAR_CHAT':
             # Render Avatar Explanation
             emotion = segment.get('avatar_pose', 'neutral')
             
             # Calculate continuous flow (Slide In only on first, Slide Out only on last)
             do_slide_in = True
             do_slide_out = True
             
             # Check Previous
             if i > 0:
                 prev = visual_plan['visual_plan'][i-1]
                 if prev.get('visual_tool') == 'AVATAR_CHAT':
                     do_slide_in = False
                     
             # Check Next
             if i < len(visual_plan['visual_plan']) - 1:
                 nxt = visual_plan['visual_plan'][i+1]
                 if nxt.get('visual_tool') == 'AVATAR_CHAT':
                     do_slide_out = False
             
             # Resolve Background Clip
             bg_clip_filename = segment.get('clip_filename')
             bg_clip_path = None
             if bg_clip_filename:
                 bg_clip_path = find_clip_path(bg_clip_filename)
             
             # FALLBACK: If no explicit background clip, or file not found, use last valid clip
             if not bg_clip_path and last_valid_clip_path:
                 bg_clip_path = last_valid_clip_path
                 st.write(f'   ⚠️ No explicit BG clip, using fallback: {os.path.basename(bg_clip_path)}')

             st.write(f'   🗣️ Rendering Avatar Chat ({emotion}) [In:{do_slide_in}, Out:{do_slide_out}] [BG: {os.path.basename(bg_clip_path) if bg_clip_path else "NONE"}]...')
             source_path = render_avatar_explanation(
                 scene_id=f"seg_{i}",
                 emotion=emotion,
                 duration=duration,
                 base_dir=movie_folder_path,
                 slide_in=do_slide_in,
                 slide_out=do_slide_out,
                 bg_video_path=bg_clip_path
             )
             seek_time = "00:00"

        if source_path and os.path.exists(source_path):
            # Scale to 95% (1216x684) and pad with white to create frame custom effect
            vf_chain = 'scale=1216:684:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1'
            
            # For ANIMATED_IMAGE, we don't need the white padding border effect as it's already generated on white
            if visual_tool == 'ANIMATED_IMAGE' or visual_tool == 'AVATAR_CHAT':
                vf_chain = 'scale=1280:720,setsar=1'

            final_cmd_inputs = ['-ss', str(seek_time), '-t', str(duration), '-i', source_path]
            
            # Special handling for Premade Clips: STRETCH if shorter, instead of looping
            if visual_tool == 'PREMADE_CLIP':
                try:
                    src_dur = get_file_duration(source_path)
                    if src_dur > 0 and src_dur < duration:
                        # Calculate stretch factor (e.g. 5s clip, 10s needed => factor 2.0)
                        # setpts=PTS*Factor slows it down
                        stretch_factor = duration / src_dur
                        # Cap stretching to avoid extreme slow-mo (e.g. max 4x)
                        if stretch_factor < 4.0:
                            vf_chain = f"setpts={stretch_factor}*PTS,{vf_chain}"
                            # Remove -ss/-t from INPUT side (because we want full file stretched)
                            final_cmd_inputs = ['-i', source_path]
                        else:
                            vf_chain = f"setpts={stretch_factor}*PTS,{vf_chain}"
                            final_cmd_inputs = ['-i', source_path]
                except:
                    pass


            # --- DYNAMIC TEXT OVERLAY ---
            # DISABLED: Now handled as post-processing step via "apply_brick_text_overlays"
            # if segment.get('dynamic_text_overlay', False) and transcript_segments:
            #     ... (old inline logic removed)

            cmd = ['ffmpeg', '-y'] + final_cmd_inputs + ['-filter_complex', vf_chain, '-t', str(duration), '-r', '24', '-c:v', 'libx264', '-preset', 'ultrafast', '-an', segment_path_abs]
        else:
            print(f'Generating placeholder for segment {i}')
            cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', f'color=c=black:s=1280x720:r=24:d={duration}', '-c:v', 'libx264', '-preset', 'ultrafast', segment_path_abs]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            st.error(f'ffmpeg error for segment {i}: {result.stderr.decode()}')
            
        if os.path.exists(segment_path_abs):
            clip_files.append(segment_path_abs)
        else:
            st.error(f'Failed to create segment {i}')
        progress_bar.progress((i + 1) / total_segments)
    with open(concat_list_path, 'w') as f:
        for clip in clip_files:
            path_fixed = clip.replace('\\', '/')
            f.write(f"file '{path_fixed}'\n")
    output_abs = os.path.abspath(output_filename)
    st.info('Rendering Master File...')
    temp_video = os.path.abspath('build/temp_video_track.mp4')
    cmd_stitch = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', temp_video]
    subprocess.run(cmd_stitch, check=True)
    # Subtitles disabled as per user request
    # srt_path_fixed = srt_path.replace('\\', '/').replace(':', '\\:')
    # cmd_final = ['ffmpeg', '-y', '-i', temp_video, '-i', audio_path, '-vf', f"subtitles='{srt_path_fixed}':force_style='Fontname=Arial,FontSize=16,PrimaryColour=&HFFFFFF,OutlineColour=&H40000000,BorderStyle=1,Outline=1,Shadow=0,MarginV=25,Alignment=2'", '-c:v', 'libx264', '-c:a', 'aac', '-map', '0:v', '-map', '1:a', output_abs]
    cmd_final = ['ffmpeg', '-y', '-i', temp_video, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v', '-map', '1:a', output_abs]
    result = subprocess.run(cmd_final, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        st.error(f'FFMPEG Error: {result.stderr.decode()}')
        return None
    return output_filename

for folder in ['assets', 'media', 'temp_frames', 'scripts']:
    os.makedirs(folder, exist_ok=True)

st.set_page_config(page_title='AI Video Engine', layout='wide', page_icon='🎬')
st.markdown('''
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .stTextArea textarea { background-color: #262730; color: #00ffcc; font-family: 'Courier New', monospace; }
    </style>
    ''', unsafe_allow_html=True)

st.sidebar.title('⚙️ Settings')

# --- LLM Source Selection ---
st.sidebar.subheader('🤖 Text Generation Model')
llm_source = st.sidebar.radio(
    'Choose LLM for script/plan generation:',
    ['🌐 Gemini API', '💻 Local Ollama'],
    index=1,
    help='Gemini API requires API key. Ollama runs locally (free, offline).'
)
use_ollama = llm_source == '💻 Local Ollama'

# Ollama model selection
selected_ollama_model = 'gemma3:4b'  # default
ollama_models = []

if use_ollama:
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=2)
        if r.status_code == 200:
            ollama_models = [m['name'] for m in r.json().get('models', [])]
            if ollama_models:
                # Prioritize good writing models
                priority_order = ['gemma3:4b', 'qwen2.5:3b', 'qwen2.5:7b', 'qwen3:4b', 'phi3:mini', 'phi3', 'llama3.2:3b', 
                                  'gemma2:2b', 'gemma2:9b', 'mistral', 'deepseek-r1:1.5b']
                
                # Sort models: priority first, then alphabetical
                def model_priority(m):
                    try:
                        return priority_order.index(m)
                    except ValueError:
                        return 100
                
                sorted_models = sorted(ollama_models, key=model_priority)
                
                selected_ollama_model = st.sidebar.selectbox(
                    '🧠 Select Ollama Model:',
                    sorted_models,
                    index=0,
                    help='Recommended: qwen2.5:3b, qwen3:4b, phi3:mini for script writing'
                )
                
                # Show model info
                st.sidebar.success(f'✅ Ollama ready: **{selected_ollama_model}**')
                
                # Size hints for common models
                size_hints = {
                    'gemma3:4b': '~2.5GB VRAM - ⭐ New! State of the art',
                    'qwen2.5:3b': '~2GB VRAM - ⭐ Best for writing',
                    'qwen3:4b': '~2.5GB VRAM - ⭐ Excellent quality',
                    'phi3:mini': '~2.3GB VRAM - Great reasoning', 
                    'llama3.2:3b': '~2GB VRAM - Good general',
                    'gemma2:2b': '~1.6GB VRAM - Fast & balanced',
                    'deepseek-r1:1.5b': '~1GB VRAM - Too small for scripts',
                    'mistral': '~4GB+ VRAM - May be slow',
                }
                hint = size_hints.get(selected_ollama_model, '')
                if hint:
                    st.sidebar.caption(hint)
                
                # Thinking mode toggle for models that support it (Qwen3, Gemma3, DeepSeek-R1)
                thinking_models = ['gemma3', 'qwen3', 'deepseek-r1']
                is_thinking_model = any(tm in selected_ollama_model.lower() for tm in thinking_models)
                
                disable_thinking = False
                if is_thinking_model:
                    disable_thinking = st.sidebar.checkbox(
                        '⚡ Disable Thinking Mode',
                        value=True,  # Default to disabled for faster output
                        help='Gemma3/Qwen3/DeepSeek-R1 have thinking mode. Disable for faster, direct responses.'
                    )
                    if disable_thinking:
                        st.sidebar.caption('💨 Fast mode: Direct output without reasoning')
                    else:
                        st.sidebar.caption('🧠 Thinking mode: Shows reasoning (slower)')
                
                # Store thinking mode setting in session_state
                st.session_state['ollama_disable_thinking'] = disable_thinking
            else:
                st.sidebar.warning('⚠️ No models installed. Try: `ollama pull qwen2.5:3b`')
        else:
            st.sidebar.error('❌ Ollama server not responding')
    except:
        st.sidebar.error('❌ Ollama not running. Start with: `ollama serve`')


st.sidebar.markdown('---')

gemini_key = st.sidebar.text_input('Gemini API Key', value=config_keys.get('gemini_api_key', ''), type='password')

if 'gemini_key_offset' not in st.session_state:
    st.session_state['gemini_key_offset'] = 0

if not use_ollama:
    if st.sidebar.button("🔄 Force Switch API Key"):
        st.session_state['gemini_key_offset'] += 1
        st.rerun()

    if gemini_key:
         key_count = len([k for k in gemini_key.split(',') if k.strip()])
         if key_count > 1:
             current_idx = (st.session_state['gemini_key_offset'] % key_count) + 1
             st.sidebar.caption(f"🔑 Active Key: {current_idx}/{key_count}")

pexels_key = st.sidebar.text_input('Pexels API Key', value=config_keys.get('pexels_api_key', ''), type='password')
st.sidebar.markdown('---')
st.sidebar.title('📁 Movie Assets')
movie_folder = st.sidebar.text_input('Movie Folder Path', value='c:\\Users\\USER\\Downloads\\Krishnawaranam\\downloads', help='Folder containing your Coraline video files')

# New Clip-based Loading
# Replaces specific single-file markdown loading
md_content = ''
parsed_segments = parse_all_clip_markdowns(movie_folder)
if parsed_segments:
    st.sidebar.success(f'✅ Auto-loaded {len(parsed_segments)} clips from Markdown files')
else:
    st.sidebar.warning('⚠️ No Clip*.md files found.')

st.sidebar.markdown('---')
st.sidebar.title('🎨 Asset Tools')
selected_char_key = st.sidebar.selectbox('Select Character to Clean', list(CHARACTER_IMAGES.keys()))
if st.sidebar.button('🪄 Test Background Removal'):
    img_filename = CHARACTER_IMAGES[selected_char_key]
    img_path = os.path.join(movie_folder, img_filename)
    
    if os.path.exists(img_path):
        st.info(f'Processing {img_filename}...')
        try:
            from rembg import remove
            original = Image.open(img_path).convert('RGBA')
            processed = remove(original)
            
            # Show side by side
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(original, caption='Original', use_container_width=True)
            with col_b:
                st.image(processed, caption='AI Cleaned (rembg)', use_container_width=True)
            
            # Save functionality
            processed_path = os.path.join(movie_folder, f"clean_{img_filename}")
            st.session_state['processed_img'] = processed
            st.session_state['processed_path'] = processed_path
            st.success('✅ Preview Generated! See above.')
            
        except ImportError:
            st.error("❌ 'rembg' library not installed. Please install it first.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error(f"Image not found: {img_path}")

if 'processed_img' in st.session_state and st.sidebar.button('💾 Save Clean Image'):
    save_path = st.session_state['processed_path']
    st.session_state['processed_img'].save(save_path)
    st.sidebar.success(f"Saved to {os.path.basename(save_path)}")
    # Update mapping to use the new file?? 
    # Maybe better to just print instructions or overwrite?
    # For now, let's just save as 'clean_...' and user can rename if they want.
    st.info(f"Saved as: {os.path.basename(save_path)}")

if 'video_topic' not in st.session_state:
    st.session_state['video_topic'] = 'The Hidden Entomology of Coraline'

video_topic = st.text_area('🧠 Video Topic / Theory (Paste full details here)', key='video_topic', height=150, help="Paste your full theory or script idea here. The AI will use this as the primary source material.")

# Video duration selector
if 'video_duration_mins' not in st.session_state:
    st.session_state['video_duration_mins'] = 3
video_duration_mins = st.number_input(
    '⏱️ Target Video Length (minutes)', 
    min_value=1, 
    max_value=15, 
    value=st.session_state['video_duration_mins'],
    step=1,
    help='Approximate duration of the final video'
)
st.session_state['video_duration_mins'] = video_duration_mins

# Initialize the appropriate LLM client based on user selection
client = None
gemini_tts_handler = None  # Separate handler for TTS - always uses Gemini API

# Always create Gemini handler for TTS if key exists
if gemini_key:
    gemini_tts_handler = GeminiHandler(key_string=gemini_key, start_index=st.session_state.get('gemini_key_offset', 0))
    # Store in session_state so generate_voiceover can access it
    st.session_state['gemini_tts_handler'] = gemini_tts_handler

# Create text generation client based on selection
if use_ollama:
    # Use local Ollama with selected model for TEXT generation only
    client = OllamaHandler(model_name=selected_ollama_model)
    if not client.connected:
        st.warning('⚠️ Ollama not connected. Start with: `ollama serve`')
        client = None
    # Show TTS info
    if gemini_tts_handler:
        st.sidebar.info('📢 TTS: Using Gemini API for voiceover')
    else:
        st.sidebar.warning('📢 TTS: Will use gTTS fallback (no Gemini key)')
elif gemini_key:
    # Use Gemini API for both text and TTS
    client = gemini_tts_handler  # Same handler for both

st.sidebar.markdown('---')
st.sidebar.subheader('🗣️ Voiceover Settings')
tts_engine = st.sidebar.radio(
    "TTS Engine:",
    ["Gemini API", "Kokoro TTS (Local)", "gTTS (Fallback)"],
    index=0
)
st.session_state['tts_engine'] = tts_engine

if tts_engine == "Kokoro TTS (Local)":
    # Check if files exist
    kokoro_ready = True
    if not os.path.exists(KOKORO_MODEL_PATH):
        st.sidebar.error("❌ kokoro-v1.0.onnx missing!")
        kokoro_ready = False
    if not os.path.exists(KOKORO_VOICES_PATH):
        st.sidebar.error("❌ voices-v1.0.bin missing!")
        kokoro_ready = False
    
    if kokoro_ready:
        st.session_state['kokoro_voice'] = st.sidebar.selectbox(
            "Voice:", 
            KOKORO_SUPPORTED_VOICES, 
            index=KOKORO_SUPPORTED_VOICES.index("af_sarah") if "af_sarah" in KOKORO_SUPPORTED_VOICES else 0
        )
        st.session_state['kokoro_speed'] = st.sidebar.slider("Speed:", 0.5, 2.0, 1.1, 0.1)
    else:
        st.sidebar.warning("Using gTTS as fallback until Kokoro files are present.")

elif tts_engine == "Gemini API":
     st.sidebar.caption("Using Google's high-quality 'Kore' voice.")


st.sidebar.markdown('---')
st.sidebar.title('💡 Video Suggestions')

if 'generated_ideas' not in st.session_state:
    st.session_state['generated_ideas'] = []

if st.sidebar.button('Generate New Govindas'):
    if not client:
        st.sidebar.error('Need API Key')
    else:
        with st.spinner('Brainstorming...'):
            try:
                import random
                import time
                
                # Use current clips to understand channel vibe
                clips_summary = "\n".join([c['description'][:100] for c in parsed_segments[:20]])
                
                # Random elements for variety
                random_seed = int(time.time() * 1000) % 100000
                categories = [
                    "hidden symbolism", "character psychology", "dark theories",
                    "animation secrets", "book vs movie", "deleted scenes mysteries",
                    "fan theories debunked", "Easter eggs", "behind the scenes horror",
                    "alternative endings", "timeline theories", "creature origins",
                    "unsolved mysteries", "hidden messages", "connections to other films"
                ]
                random_categories = random.sample(categories, 3)
                
                # Track previous ideas to avoid repetition
                previous_titles = [idea.get('title', '') for idea in st.session_state.get('generated_ideas', [])]
                avoid_text = f"\n\nAVOID repeating these previous ideas: {previous_titles}" if previous_titles else ""
                
                prompt = f"""
                SEED: {random_seed}
                
                Generate 5 viral YouTube Video Essay ideas for a detailed Coraline theory channel.
                
                STYLE REFERENCE: Karsten Runquist, The Theorizer, MatPat.
                The titles must be SHORT, PUNCHY, and visually evocative.
                
                FORCE one of these EXACT title formats for each idea:
                1. "The [Entity] Theory" (e.g. "The Bug Theory", "The Door Theory")
                2. "How I'd Survive [Event]" (e.g. "How I'd Survive The Other World")
                3. "[Statement of Fact]" (e.g. "The Door Is Fake", "She Never Escaped")
                4. "[Character]'s [Hidden Trait]" (e.g. "Coraline's Parents Knew")
                5. "They [Action]" (e.g. "They Knew", "They Lied")
                
                TITLING RULES:
                - MAX 6 WORDS. Shorter is better.
                - Use lowercase aesthetic or simple forceful statements.
                - NO colon subtitles (e.g. NOT "Coraline: The Dark Truth").
                - NO generic adjectives like "Hidden", "Dark", "Secret" unless part of a specific theory name.
                
                Based on clips summary:
                {clips_summary}
                {avoid_text}
                
                Return JSON list:
                - "title": The precise title following patterns above.
                - "thumbnail": Visual concept. description: "High contrast, big bold text saying 'FAKE', arrow pointing to door", etc.
                - "hook": The first sentence of the script. Must be a direct challenge to the viewer.
                """
                resp = client.generate_content(contents=prompt)
                text = resp.text.strip()
                if text.startswith('```json'): text = text[7:]
                if text.startswith('```'): text = text[3:]
                if text.endswith('```'): text = text[:-3]
                st.session_state['generated_ideas'] = json.loads(text)
            except Exception as e:
                st.sidebar.error(f'Error: {e}')

def set_topic(new_topic):
    st.session_state['video_topic'] = new_topic

if st.session_state['generated_ideas']:
    for i, idea in enumerate(st.session_state['generated_ideas']):
        with st.sidebar.expander(f"💡 {idea['title']}"):
            st.caption(f"**Hook:** {idea['hook']}")
            st.caption(f"**Thumbnail:** {idea['thumbnail']}")
            st.button(f"Use This Theory", key=f"use_idea_{i}", on_click=set_topic, args=(idea['title'],))

SCENE_DATA = {'title': 'The Hidden Entomology of Coraline', 'scenes': [{'id': 1, 'duration': 7, 'visual_tool': 'EXTERNAL_MOVIE_CLIP', 'description': 'Coraline bath bugs', 'code_or_prompt': 'Clip from Coraline (2009) bath scene'}, {'id': 2, 'duration': 8, 'visual_tool': 'MANIM', 'description': 'Bug Theory Intro', 'code_or_prompt': 'from manim import *\nclass BugTheory(Scene):\n    def construct(self):\n        title = Text(\'THE BUG THEORY\', font_size=72, color=YELLOW)\n        self.play(Write(title))\n        self.wait(2)\n        self.play(FadeOut(title))'}, {'id': 3, 'duration': 5, 'visual_tool': 'EXTERNAL_MOVIE_CLIP', 'description': 'Dragonfly ornaments', 'code_or_prompt': 'Clip of dragonfly ornaments'}, {'id': 5, 'duration': 8, 'visual_tool': 'EXTERNAL_MOVIE_CLIP', 'description': 'Beldam sewing', 'code_or_prompt': 'Opening credits sewing scene'}, {'id': 8, 'duration': 10, 'visual_tool': 'MANIM', 'description': 'Mantis Logic', 'code_or_prompt': 'from manim import *\nclass MantisLogic(Scene):\n    def construct(self):\n        m = Text(\'Mantis\'); w = Text(\'Wisdom\'); c = Text(\'Control\', color=RED)\n        arr1 = Arrow(LEFT, RIGHT); arr2 = Arrow(LEFT, RIGHT)\n        group = VGroup(m, arr1, w, arr2, c).arrange(RIGHT)\n        self.play(Write(m), Write(arr1), Write(w))\n        self.wait(1)\n        self.play(ReplacementTransform(w, c))\n        self.wait(1)'}]}

if 'scripts' not in st.session_state:
    st.session_state['scripts'] = {}
if 'clip_paths' not in st.session_state:
    st.session_state['clip_paths'] = {}
if 'generated_scene_data' not in st.session_state:
    st.session_state['generated_scene_data'] = None
if 'parsed_segments' not in st.session_state:
    st.session_state['parsed_segments'] = []
if parsed_segments:
    st.session_state['parsed_segments'] = parsed_segments

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

def generate_voiceover(handler, scene_id, text):
    """
    Generates voiceover audio using selected TTS engine (Gemini, Kokoro, or gTTS).
    """
    engine = st.session_state.get('tts_engine', 'Gemini API')
    path = f'media/vo_{scene_id}.wav'
    
    # --- KOKORO TTS ---
    if engine == "Kokoro TTS (Local)":
        if os.path.exists(KOKORO_MODEL_PATH) and os.path.exists(KOKORO_VOICES_PATH):
            try:
                voice = st.session_state.get('kokoro_voice', 'af_sarah')
                speed = st.session_state.get('kokoro_speed', 1.0)
                
                # Temp input file
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt", encoding='utf-8') as temp_input_file:
                    temp_input_file.write(text)
                    temp_input_path = temp_input_file.name

                path_abs = os.path.abspath(path)
                
                # Command
                command = [
                    "kokoro-tts",
                    temp_input_path,
                    path_abs,
                    "--lang", "en-us",
                    "--voice", voice,
                    "--speed", str(speed),
                    # "--format", "wav", # kokoro-tts might auto-detect or default to wav if extension is wav
                    # User's snippet used .mp3 and --format mp3. Here we want wav generally.
                    # python-kokoro-tts usually saves as wav by default or based on extensions.
                    # Let's try explicitly if supported or just let it infer.
                    # Checking user snippet: "--format", "mp3".
                    # We want .wav for our pipeline.
                    "--model", KOKORO_MODEL_PATH,
                    "--voices", KOKORO_VOICES_PATH
                ]
                
                # Run
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
                
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    check=False,
                    env=env,
                    shell=True 
                )
                
                # Cleanup input
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                    
                if result.returncode == 0 and os.path.exists(path_abs) and os.path.getsize(path_abs) > 0:
                    return path
                else:
                    st.warning(f"Kokoro TTS failed (Code {result.returncode}). Stderr: {result.stderr[:200]}")
                    # Fallback to gTTS below
            except Exception as e:
                st.warning(f"Kokoro TTS Exception: {e}")
        else:
            st.warning("Kokoro model files missing. Falling back...")

    # --- GEMINI TTS ---
    elif engine == "Gemini API":
        tts_handler = st.session_state.get('gemini_tts_handler', None)
        if tts_handler:
            try:
                response = tts_handler.generate_content(
                    model='gemini-2.5-flash-preview-tts', 
                    contents=text, 
                    config=types.GenerateContentConfig(
                        response_modalities=['AUDIO'], 
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore')
                            )
                        )
                    )
                )
                data = response.candidates[0].content.parts[0].inline_data.data
                wave_file(path, data)
                return path
            except Exception as e:
                st.warning(f'Gemini TTS failed: {e}. Using fallback.')
    
    # --- FALLBACK: gTTS ---
    path_mp3 = f'media/vo_{scene_id}.mp3'
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(path_mp3)
        return path_mp3
    except Exception as e:
        st.error(f"gTTS Failed: {e}")
        return None

def render_manim(scene_id, code):
    script_path = f'scripts/scene_{scene_id}.py'
    with open(script_path, 'w') as f:
        f.write(code)
    class_name = 'Scene'
    if 'class ' in code:
        class_name = code.split('class ')[1].split('(')[0].strip()
    output_dir = os.path.abspath('media')
    try:
        cmd = ['manim', '-qm', '--media_dir', output_dir, script_path, class_name, '--format=mp4']
        subprocess.run(cmd, check=True)
        found_file = list(Path(output_dir).rglob(f'{class_name}.mp4'))
        if found_file:
            final_path = f'media/manim_{scene_id}.mp4'
            shutil.move(str(found_file[0]), final_path)
            return final_path
    except Exception as e:
        st.error(f'Manim Error: {e}')
    return None

def render_title_card(scene_id, title_text, duration=3):
    """
    Renders a beautiful title card using HTML/CSS + Selenium.
    """
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700&family=Lato:wght@300&display=swap');
            body {{
                margin: 0;
                padding: 0;
                width: 1280px;
                height: 720px;
                background-color: #0e1117;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
            }}
            .container {{
                text-align: center;
                max-width: 90%;
                animation: fadeIn 1s ease-in;
            }}
            h1 {{
                font-family: 'Cinzel', serif;
                font-size: 80px;
                color: #f0f0f0;
                margin: 0;
                text-transform: uppercase;
                letter-spacing: 10px;
                line-height: 1.2;
                word-wrap: break-word; /* Ensure long words break */
                text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
                animation: scaleUp 4s ease-out forwards;
            }}
            /* Dynamically adjust font size for very long titles if needed handled by CSS clamp or media queries? 
               For now, simple max-width and wrapping should suffice. */
            
            .line {{
                width: 0%;
                height: 2px;
                background: linear-gradient(90deg, transparent, #ff4b4b, transparent);
                margin: 20px auto;
                animation: expandLine 1.5s ease-out forwards 0.5s;
            }}
            @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
            @keyframes scaleUp {{ from {{ transform: scale(0.9); }} to {{ transform: scale(1.0); }} }} /* Reducing scale to avoid overflow */
            @keyframes expandLine {{ to {{ width: 60%; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title_text}</h1>
            <div class="line"></div>
        </div>
        <script>
             // Auto-scale font if too large
             const h1 = document.querySelector('h1');
             if (h1.innerText.length > 20) {{
                 h1.style.fontSize = '60px';
             }}
             if (h1.innerText.length > 40) {{
                 h1.style.fontSize = '40px';
             }}
        </script>
    </body>
    </html>
    '''
    
    html_path = f'temp_title_{scene_id}.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--window-size=1280,720')
    options.add_argument('--hide-scrollbars')
    options.add_argument('--force-device-scale-factor=1') # Ensure consistent rendering
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    temp_dir = f'temp_frames/title_{scene_id}'
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        driver.get(Path(html_path).absolute().as_uri())
        # Wait for fonts/render and animation start
        time.sleep(1.0) 
        
        # Capture frame for static image + pan/zoom
        # Wait a bit longer to let animation settle near end state or mid state
        time.sleep(1.0) 
        output_png = f'media/title_card_{scene_id}.png'
        driver.save_screenshot(output_png)
        
        # Create a video from this image with a subtle zoom and PAN effect to make it dynamic
        output_mp4 = f'media/title_{scene_id}.mp4'
        
        # Zoom effect: zoom 1.0 to 1.1 over duration
        # We start with the captured image and just zoom slightly.
        cmd = [
            'ffmpeg', '-y', '-loop', '1', '-i', output_png, 
            '-vf', f"zoompan=z='min(zoom+0.0005,1.15)':d={int(duration*25)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',scale=1280:720,setsar=1", 
            '-c:v', 'libx264', '-t', str(duration), '-pix_fmt', 'yuv420p', 
            '-r', '24', output_mp4
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return output_mp4
        
    except Exception as e:
        print(f"Title Render Error: {e}")
        return None
    finally:
        driver.quit()
        if os.path.exists(html_path):
            os.remove(html_path)

def render_threejs(scene_id, p5_code, duration=10):
    html_content = f'''
    <html>
    <head><script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/p5.js"></script></head>
    <body style="margin:0; overflow:hidden; background:black;">
        <script>
        {p5_code}
        new p5(sketch);
        window.nextFrame = () => {{ }}; 
        </script>
    </body>
    </html>
    '''
    html_path = f'temp_sketch_{scene_id}.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--window-size=1280,720')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        driver.get(f'file://{os.path.abspath(html_path)}')
        time.sleep(3)
        temp_dir = f'temp_frames/scene_{scene_id}'
        os.makedirs(temp_dir, exist_ok=True)
        frames = int(duration * 30)
        progress = st.progress(0)
        for i in range(frames):
            driver.save_screenshot(f'{temp_dir}/frame_{i:04d}.png')
            progress.progress((i + 1) / frames)
        from moviepy.editor import ImageSequenceClip
        output_file = f'media/threejs_{scene_id}.mp4'
        clip = ImageSequenceClip(temp_dir, fps=30)
        clip.write_videofile(output_file, fps=30, codec='libx264')
        return output_file
    finally:
        driver.quit()
        if os.path.exists(html_path):
            os.remove(html_path)

def fetch_pexels_video(query, scene_id):
    headers = {'Authorization': pexels_key}
    url = f'https://api.pexels.com/videos/search?query={query}&per_page=1'
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        video_url = r.json()['videos'][0]['video_files'][0]['link']
        path = f'media/pexels_{scene_id}.mp4'
        with requests.get(video_url, stream=True) as v_file, open(path, 'wb') as f:
            shutil.copyfileobj(v_file.raw, f)
        return path
    return None

st.title('🎬 Coraline Theory Generator')

# --- PHASE 0: ADVANCED TRAILER PREP ---
with st.expander("🎬 PHASE 0: Advanced Trailer Scene Prep (Auto-Theory)", expanded=False):
    st.info("Download and analyze a movie trailer to build a custom scene database.")
    t_movie_name = st.text_input("Movie Name", value="Coraline")
    t_manual_url = st.text_input("Manual Trailer URL (Optional)", value="")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("🚀 Download Trailer"):
            with st.spinner("Downloading trailer..."):
                t_path = download_trailer(t_movie_name, manual_url=t_manual_url, handler=client)
                if t_path:
                    st.success(f"Trailer ready: {t_path}")
                    st.video(t_path)
                    st.session_state['active_trailer_path'] = t_path
    with col_t2:
        if st.button("🔍 Analyze & Catalog Scenes"):
            t_path = st.session_state.get('active_trailer_path')
            if not t_path:
                st.error("Download trailer first!")
            else:
                with st.spinner("Detecting scenes and generating Gemini descriptions..."):
                    cat = catalog_trailer_scenes(t_path, t_movie_name, handler=client)
                    if cat:
                        st.session_state['auto_scene_catalog'] = cat
                        st.success(f"Successfully cataloged {len(cat)} scenes!")
    
    if st.session_state.get('auto_scene_catalog'):
        st.subheader("📋 Scene Catalog")
        cat_df = pd.DataFrame(st.session_state['auto_scene_catalog'])
        st.dataframe(cat_df[['id', 'start', 'end', 'duration', 'description']], hide_index=True)

if st.session_state.get('parsed_segments'):
    with st.expander(f"📚 Loaded {len(st.session_state['parsed_segments'])} Pre-made Clips"):
        for seg in st.session_state['parsed_segments'][:10]:
            st.markdown(f"**🎬 {seg['filename']}**")
            st.caption(f"{seg['description'][:200]}...")
        if len(st.session_state['parsed_segments']) > 10:
            st.info(f"...and {len(st.session_state['parsed_segments']) - 10} more clips")

st.markdown('---')
st.subheader('📻 The Radio Edit Workflow')
if 'radio_script' not in st.session_state:
    st.session_state['radio_script'] = """Coraline isn't just a beautifully animated dark fantasy; it's a masterclass in subtle, bone-chilling foreshadowing. Every rewatch peels back another layer, revealing details that hint at the terror to come, long before Coraline ever steps through that tiny door. It's like Laika built a whole secret language into the animation itself. And today, I want to talk about some of those truly terrifying breadcrumbs, those little moments that, once you see them, you can never unsee. They reshape how you experience Coraline, transforming it from a dark fairy tale into something far more insidious, showing just how long the Beldam has been watching, waiting.

## The Stitched Sentinel
From the very beginning, the Other Mother has a spy in Coraline’s world. We see the doll, painstakingly crafted to resemble Coraline, even down to her signature blue hair. It’s introduced early, lying on the floor [Clip_01_Scene_video-Scene-024, 035, 050], seeming like an innocent toy. But this doll is not just a precursor; it *is* the Other Mother's eyes and ears. It’s the very tool she uses to lure Coraline into her fabricated world. Notice how Coraline often holds it in her real bedroom, looking uneasy [Clip_01_Scene_video-Scene-065, 068, 070, 075]. It's as if a part of the Other World has already seeped into hers, observing her vulnerabilities and desires. This detail haunted me for weeks after I first connected it.

## Decay and Distraction
Coraline’s real home, the Pink Palace, is depicted as dreary and neglected. We see her father, constantly hunched over his computer, oblivious to Coraline's attempts to connect [Clip_01_Scene_video-Scene-001, 002, 003, 018]. Her mother is equally engrossed, often irritable [Clip_01_Scene_video-Scene-043, 045, 062]. This neglect is a terrifying detail because it’s the very crack the Beldam exploits. The rundown house, the "twelve leaky windows" Coraline lists [Clip_01_Scene_video-Scene-006], the "disgusting bugs" [Clip_01_Scene_video-Scene-016] in her bathroom – these aren't just mundane inconveniences. They are symptoms of the real world's apathy, creating a vacuum that the Other Mother's vibrant, perfect world is designed to fill. Even the Other Father is shown typing on a similar computer [Clip_01_Scene_video-Scene-038, 040], echoing Coraline's real-world frustration, but with a sinister, manufactured focus.

## The Prismatic Prison
This is where it gets really chilling. The snow globes. Early on, we see Coraline’s parents' treasured Detroit Zoo snow globe [Clip_01_Scene_video-Scene-027]. It seems like a memento, a nostalgic piece of their past life. But later, in the Other World, the Other Mother *gives* Coraline a snow globe, holding her close in a creepy, intimate gesture [Clip_01_Scene_video-Scene-028]. It's a subtle but horrifying clue. Then, when Coraline's *real* parents are trapped, where are they? Inside a snow globe [Clip_10_Scene_0000, 0002]! The film lays it out for us, showing the very mechanism of their capture within something so beautiful and fragile. The snow globe isn't just a toy; it's a symbolic prison, and the Other Mother introduces it to Coraline herself.

## Echoes from the Lost Souls
The appearance of the Ghost Children is, for me, the most direct and heartbreaking foreshadowing. They materialize in Coraline's dream, ethereal and vulnerable, to warn her [Clip_10_Scene_0027, 0029, 0030]. They describe being lured, played with, and then having their eyes replaced with buttons, just like the Other Mother intends for Coraline [Clip_10_Scene_0034]. Their golden, angelic forms against a swirling night sky are beautiful yet deeply unsettling [Clip_10_Scene_0035, 0036, 0037]. They are literal echoes of the Beldam's past victims, and their story is Coraline’s future if she fails. Laika’s choice to show their ultimate horror so plainly, yet in such a dreamlike way, is masterful. It’s a terrifying truth delivered as a whispered warning.

When you piece all these seemingly disparate details together – the doll as a scout, the real world's decay as an invitation, the snow globes as miniature prisons, and the ghost children as grim prophets – you realize that the Beldam isn’t just reacting to Coraline. She's been planning this, refining her trap, for generations. She's a meticulous predator, and these details reveal the chilling extent of her patient, terrifying game. Every single one of these moments is a tiny piece of a larger, much older, and truly terrifying design. And that, for me, is the enduring genius of Coraline."""
if 'radio_audio' not in st.session_state:
    st.session_state['radio_audio'] = None
if 'radio_segments' not in st.session_state:
    st.session_state['radio_segments'] = None
if 'radio_visual_plan' not in st.session_state:
    st.session_state['radio_visual_plan'] = None

col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
with col_r1:
    use_clips_context = st.checkbox('Use Clip Context', value=True, help="Uncheck to write script without attaching specific clip data.")
    if st.button('1️⃣ Generate Full Voiceover Script'):
        if not client:
            st.error('Need API Key')
        elif use_clips_context and not st.session_state['parsed_segments']:
            st.warning('⚠️ Please load the Movie Markdown file first (Sidebar or Auto-load).')
        else:
            with st.spinner('Writing complete, word-for-word voiceover script...'):
                if use_clips_context:
                    movie_context = json.dumps(st.session_state['parsed_segments'], indent=2)
                    evidence_section = f"MOVIE EVIDENCE (Use this as your foundation):\n{movie_context}"
                    rules_section = "- You must use the MOVIE EVIDENCE to insure your points are valid and supported by actual footage."
                else:
                    evidence_section = "MOVIE EVIDENCE: None provided. Use your detailed internal knowledge of the film."
                    rules_section = "- Ensure your points are accurate to the film's events."
                
                target_mins = st.session_state.get('video_duration_mins', 3)
                # Estimate word count: ~150 words per minute for natural speech
                target_words = target_mins * 150
                prompt = f'''
                 You are a dark, investigative storyteller. You are NOT a film critic. You are NOT a reviewer.
                 
                 YOUR GOAL: To adapt the provided "THEORY CONTENT" into a compelling, spoken narrative script. 
                 
                 THEORY CONTENT (SOURCE MATERIAL):
                 """
                 {video_topic}
                 """
                 
                 **TARGET LENGTH: {target_mins} minutes (approximately {target_words} words)**
                 
                 {evidence_section}
                 
                 CRITICAL INSTRUCTIONS:
                 1. **FOCUS ON THE THEORY:** Your script must be 100% focused on explaining the theory and the evidence provided in the "THEORY CONTENT".
                 2. **NO FILM CRITICISM:** Do NOT praise the animation, the studio (Laika), or the director. Do NOT use words like "masterpiece", "brilliant", "stunning", "gem", "classic". We are here to discuss the LORE, not the movie's production quality.
                 3. **STORYTELLING TONE:** Treat the movie events as REAL. Speak like a detective connecting clues. "The Beldam set a trap..." instead of "The writers wrote a trap...".
                 4. **EXPAND INTELLIGENTLY:** If the provided content is short, expand upon it using the MOVIE EVIDENCE provided above, but stay within the theme of the theory.
                 
                 VOICE & PERSONALITY:
                 - **Serious & Immersive:** You are deep in the rabbit hole. You sound slightly unsettled by what you've found.
                 - **Direct:** Get straight to the point. No fluff.
                 - **"We" vs "I":** Use "We see..." or "Notice how..." to guide the viewer.
                 
                 STRUCTURE:
                 1. **The Hook:** Start immediately with the core disturbing idea. No "Hello guys".
                 2. **Chapters:** Use '## Chapter Name' to divide valid sections.
                 3. **The Conclusion:** A final haunting thought based on the theory.
                 
                 STRICT OUTPUT RULES:
                 {rules_section}
                 - NEVER references filenames/timestamps.
                 - NO generic praise ("This film is great").
                 - NO YouTube intros/outros ("Like and subscribe").
                 - USE '## ' to denote chapter headers.
                 '''
                try:
                    resp = client.generate_content(contents=prompt)
                    st.session_state['radio_script'] = resp.text
                    st.success('Full Script Generated! Review in the editor below.')
                except Exception as e:
                    st.error(f'Generation failed: {e}')

    uploaded_vos = st.file_uploader("🎙️ Manual VO Upload", type=['wav', 'mp3', 'm4a'], accept_multiple_files=True, help="Upload one or more audio chunks. They will be merged in order (sorted by filename).")
    if st.button('2️⃣ Analyze Timing'):
        if uploaded_vos:
             with st.spinner('Processing Uploaded Audio & Analyzing with Whisper...'):
                 # Sort to ensure chunks are in order (part_1, part_2, etc)
                 uploaded_vos.sort(key=lambda x: x.name)
                 
                 os.makedirs('media', exist_ok=True)
                 os.makedirs('build/temp_chunks', exist_ok=True)
                 
                 chunk_paths = []
                 for i, uploaded_vo in enumerate(uploaded_vos):
                     # Save uploaded file
                     ext = os.path.splitext(uploaded_vo.name)[1]
                     temp_chunk = os.path.join('build/temp_chunks', f'chunk_{i}{ext}')
                     with open(temp_chunk, 'wb') as f:
                         f.write(uploaded_vo.getbuffer())
                     
                     # Standardize format for concatenation (24kHz, Mono, WAV)
                     std_chunk = os.path.join('build/temp_chunks', f'chunk_{i}.wav')
                     cmd_std = ['ffmpeg', '-y', '-i', temp_chunk, '-ar', '24000', '-ac', '1', std_chunk]
                     subprocess.run(cmd_std, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                     chunk_paths.append(std_chunk)
                 
                 # Final master path
                 audio_path = os.path.join('media', 'full_narrative.wav')
                 
                 if len(chunk_paths) > 1:
                     st.write(f"🔗 Merging {len(chunk_paths)} chunks...")
                     list_path = os.path.join('build/temp_chunks', 'files.txt')
                     with open(list_path, 'w') as f:
                         for p in chunk_paths:
                             # Use absolute path and forward slashes for ffmpeg concat demuxer
                             clean_p = os.path.abspath(p).replace('\\', '/')
                             f.write(f"file '{clean_p}'\n")
                     
                     cmd_concat = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', audio_path]
                     subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                 else:
                     shutil.copy(chunk_paths[0], audio_path)
                 
                 if not os.path.exists(audio_path):
                     st.error("❌ Audio conversion/merge failed!")
                 else:
                     # Transcribe
                     st.write("📝 Transcribing with Whisper...")
                     model = load_whisper()
                     # Use script as initial prompt if available for better accuracy
                     initial_prompt = st.session_state.get('radio_script', '')[:1000]
                     result = model.transcribe(audio_path, word_timestamps=True, initial_prompt=initial_prompt)
                     
                     timed_segments = []
                     for segment in result['segments']:
                         seg_words = []
                         if 'words' in segment:
                             for w in segment['words']:
                                 seg_words.append({
                                     'word': w['word'],
                                     'start': w['start'],
                                     'end': w['end']
                                 })
                         
                         timed_segments.append({
                             'id': f"uploaded_{segment['id']}",
                             'start': segment['start'],
                             'end': segment['end'],
                             'text': segment['text'].strip(),
                             'visual_tool': 'PREMADE_CLIP',
                             'words': seg_words
                         })
                     
                     st.session_state['radio_audio'] = audio_path
                     st.session_state['radio_segments'] = timed_segments
                     
                     # Cleanup temp chunks
                     shutil.rmtree('build/temp_chunks')
                     st.success('✅ Uploaded Audio Chunks Merged & Analyzed!')
        else:
            if not st.session_state['radio_script']:
                st.warning('Draft script first!')
            elif not client:
                st.error('Need API Key')
            else:
                with st.spinner('Generating Audio & Analyzing with Whisper...'):
                    audio_path, timed_segments = create_radio_edit(client, st.session_state['radio_script'])
                    st.session_state['radio_audio'] = audio_path
                    st.session_state['radio_segments'] = timed_segments
                    st.success('Timing Analyzed!')

with col_r3:
    st.write("**3️⃣ Plan Visuals**")
    if st.button('🎯 Semantic Match Visuals', help="Uses Sentence Transformers to match script text to the best trailer scenes."):
        if not st.session_state.get('radio_segments'):
            st.warning('Analyze timing first!')
        elif not st.session_state.get('auto_scene_catalog'):
            st.warning('Please run Phase 0 (Trailer Prep) first to build a scene catalog.')
        else:
            with st.spinner('Semantically matching scenes to script...'):
                visual_plan = generate_visual_plan_semantic_match(
                    st.session_state['radio_segments'], 
                    st.session_state['auto_scene_catalog'], 
                    t_movie_name, 
                    handler=client
                )
                st.session_state['radio_visual_plan'] = {'visual_plan': visual_plan}
                st.success('Visuals Planned!')
    
    if st.button('🧠 Gemini Plan (Legacy)', help="Uses the original Gemini-based matching logic."):
        if not st.session_state['radio_segments']:
            st.warning('Analyze timing first!')
        elif not st.session_state['parsed_segments']:
            st.warning('Load Movie Context first!')
        else:
            with st.spinner('Gemini is matching clips to audio...'):
                visual_plan = generate_visual_plan_from_audio(client, st.session_state['radio_segments'], st.session_state['parsed_segments'])
                st.session_state['radio_visual_plan'] = visual_plan
                st.success('Visuals Planned!')

with col_r4:
    if st.button('4️⃣ Assemble Final Video'):
        if not st.session_state['radio_visual_plan']:
            st.warning('Plan visuals first!')
        else:
            with st.spinner('Assembling Final Cut...'):
                out = assemble_video_from_plan(st.session_state['radio_audio'], st.session_state['radio_visual_plan'], movie_folder, st.session_state['radio_segments'])
                if out:
                    st.session_state['rendered_video'] = out
                    st.success(f'Render Complete: {out}')
                    st.video(out)

with col_r5:
    if st.button('5️⃣ Add Text Overlay'):
        # Always use the BASE rendered video, not a previously overlaid one
        base_video = 'final_theory.mp4'
        if not os.path.exists(base_video):
            st.warning('Render video first! (final_theory.mp4 not found)')
        elif not st.session_state['radio_audio']:
            st.warning('Need audio track!')
        elif not st.session_state['radio_script']:
            st.warning('Need original script!')
        else:
            with st.spinner('Applying brick-style text overlays...'):
                final_with_text = apply_brick_text_overlays(
                    client,
                    base_video,  # Always use the base render
                    st.session_state['radio_audio'],
                    st.session_state['radio_script'],
                    output_path='final_with_text.mp4'
                )
                if final_with_text:
                    st.session_state['overlaid_video'] = final_with_text
                    st.success(f'Text overlay applied! {final_with_text}')
                    st.video(final_with_text)

if st.session_state['radio_script']:
    st.subheader('📝 Script Editor')
    st.session_state['radio_script'] = st.text_area('Edit your script here before generating audio:', st.session_state['radio_script'], height=200)

if st.session_state['radio_segments']:
    st.subheader('⏱️ Audio Timeline')
    st.audio(st.session_state['radio_audio'])
    df = pd.DataFrame(st.session_state['radio_segments'])
    st.dataframe(df[['start', 'end', 'text']])

# --- MANUAL INPUT SECTION ---
st.markdown('---')
st.subheader('🔧 Manual Input Bypass (Skip API Calls)')
st.caption('Use these if the API is slow/broken. Steps 1, 3, and 5 use Gemini API and have prompts you can copy. Step 2 uses Whisper locally. Step 4 uses FFMPEG locally.')

# ============================================
# STEP 1 BYPASS - SCRIPT MANUAL INPUT
# ============================================
with st.expander('📝 Step 1 Bypass: Script - Manual Input', expanded=False):
    st.write('**This bypasses the "Generate Full Voiceover Script" API call.**')
    
    # Generate and show the prompt that would be sent to Gemini
    st.markdown('---')
    st.write('**📋 Step A: Copy this prompt and paste it into Gemini/ChatGPT:**')
    
    if st.session_state.get('parsed_segments') and use_clips_context:
        movie_context = json.dumps(st.session_state['parsed_segments'], indent=2)
        evidence_section = f"MOVIE EVIDENCE (Use this as your foundation):\n{movie_context}"
        rules_section = "- You must use the MOVIE EVIDENCE to insure your points are valid and supported by actual footage."
    else:
        evidence_section = "MOVIE EVIDENCE: None provided. Use your detailed internal knowledge of the film."
        rules_section = "- Ensure your points are accurate to the film's events."
    
    target_mins = st.session_state.get('video_duration_mins', 3)
    target_words = target_mins * 150
    
    script_prompt = f'''You are a passionate film analyst who genuinely loves Coraline. You're not a typical "YouTuber" - you're someone who has spent countless hours studying this film because it genuinely fascinates you. Your viewers aren't just subscribers; they're people who share your obsession with uncovering hidden details.

THEORY TOPIC: "{video_topic}"

**TARGET LENGTH: {target_mins} minutes (approximately {target_words} words)**
- Write a script that, when spoken naturally, will last about {target_mins} minutes
- Adjust depth and number of points to fit this duration

{evidence_section}

VOICE & PERSONALITY:
- **Authentic Wonder:** You genuinely find this stuff fascinating. Let that curiosity show. "I kept rewatching this scene because something felt off..." or "This detail haunted me for weeks..."
- **Personal Connection:** Share brief moments of your journey. "The first time I noticed this, I had to pause the film..." or "I almost missed this, but then..."  
- **Thoughtful, Not Performative:** Avoid generic YouTube energy. No "Hey guys!" or "Smash that like button!" You speak like someone sharing a discovery with a close friend over coffee.
- **Intellectual Humility:** You're confident in your research, but acknowledge the craft. "Laika's attention to detail is insane here..." or "The animators clearly intended..."
- **Emotional Honesty:** If something is creepy, unsettling, or beautiful - say so genuinely. "This moment genuinely gives me chills" is more powerful than manufactured excitement.

NATURAL SPEECH PATTERNS:
- Use thoughtful pauses: "And then... I realized something."
- Rhetorical questions: "But why would they include this detail? What purpose does it serve?"
- Genuine reactions: "This is where it gets really interesting..." or "Now, stay with me here..."
- Avoid: "Alright folks", "Hey everyone", "What's up guys", or any generic opener

STRUCTURE:
1. **The Hook / Intro:** Start IMMEDIATELY with your opening hook. Do **NOT** use a '##' header for this first section. Just start speaking.

2. **Chapters:** After the intro (approx 30s-1min), start using headers for the deep dive:
   ## Chapter Title
   (Script content...)

3. **The Revelation:** Connect the pieces.
4. **The Reflection:** End with genuine thought.

IMPORTANT: You MUST use '## ' to denote chapter titles. But do NOT use it for the very first opening lines.

STRICT OUTPUT RULES (CRITICAL):
{rules_section}
- HOWEVER, you must NEVER, EVER include the filenames, clip IDs, or bracketed timestamps (e.g. [Clip_XX], [Timestamp]) in your output script.
- The Script MUST be pure spoken text only. Imagine you are reading this script into a microphone. You would not say "Open bracket Clip One close bracket".
- If you want to reference a scene, describe it: "We see Coraline walking..." instead of "In Clip 1 we see..."

FORBIDDEN:
- NO metadata references (filenames like "Clip_01_Scene...", timestamps like "00:23")
- NO bracketed text of ANY kind: No [Clip...], No [laughs], No [pause].
- NO call-to-actions (subscribe, like, comment)
- NO clickbait phrases ("You won't BELIEVE", "SHOCKING discovery")

Write the complete spoken script. Every word should sound like it comes from someone who truly cares about this film and respects their audience's intelligence.
IMPORTANT: You MUST use '## ' to denote chapter titles.
'''
    
    col_prompt1, col_prompt2 = st.columns([3, 1])
    with col_prompt1:
        st.text_area('📋 Script Generation Prompt:', script_prompt, height=200, key='step1_prompt_display')
    with col_prompt2:
        st.download_button(
            label="⬇️ Download Prompt",
            data=script_prompt,
            file_name="step1_script_prompt.txt",
            mime="text/plain",
            key='download_step1_prompt'
        )
    
    st.markdown('---')
    st.write('**📥 Step B: Paste the AI response (your script) here:**')
    manual_script = st.text_area('Script Text:', height=200, key='manual_script_input', placeholder='Paste the full voiceover script that Gemini/ChatGPT generated...\n\n## Chapter Title\nYour script text here...')
    
    if st.button('✅ Use Manually Entered Script', key='use_manual_script'):
        if manual_script.strip():
            st.session_state['radio_script'] = manual_script.strip()
            st.success('✅ Script loaded! You can now proceed to Step 2 (Analyze Timing).')
            st.rerun()
        else:
            st.warning('Please paste your script first.')


# ============================================
# STEP 2 BYPASS - TIMING SEGMENTS MANUAL INPUT  
# ============================================
with st.expander('⏱️ Step 2 Bypass: Timing Segments - Manual Input', expanded=False):
    st.write('**If you already have timed segments (bypasses Whisper analysis):**')
    st.caption('Format: List of objects with id, start, end, text, visual_tool, words')
    
    st.write('**Option 1: Upload JSON File**')
    uploaded_segments = st.file_uploader('Upload Segments JSON:', type=['json'], key='upload_segments')
    manual_audio_path = st.text_input('Audio File Path (required):', placeholder='e.g., build/audio_chunks/part_0_raw.wav', key='manual_audio_path')
    
    if uploaded_segments is not None:
        if st.button('✅ Load Uploaded Segments', key='load_uploaded_segments'):
            if manual_audio_path.strip():
                try:
                    content = uploaded_segments.read().decode('utf-8')
                    parsed_segments = json.loads(content)
                    st.session_state['radio_segments'] = parsed_segments
                    st.session_state['radio_audio'] = manual_audio_path.strip()
                    st.success('✅ Segments loaded from file! You can now proceed to Step 3 (Plan Visuals).')
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f'❌ Invalid JSON in file: {e}')
            else:
                st.warning('Please provide audio file path.')
    
    st.write('**Option 2: Paste JSON Text**')
    manual_segments_json = st.text_area('Segments JSON:', height=150, key='manual_segments_input', placeholder='[{"id": "txt_0_0", "start": 0.0, "end": 5.0, "text": "...", "visual_tool": "PREMADE_CLIP", "words": [...]}]')
    
    if st.button('✅ Use Pasted Segments', key='use_manual_segments'):
        if manual_segments_json.strip() and manual_audio_path.strip():
            try:
                clean_json = manual_segments_json.strip()
                if clean_json.startswith('```json'):
                    clean_json = clean_json[7:]
                if clean_json.startswith('```'):
                    clean_json = clean_json[3:]
                if clean_json.endswith('```'):
                    clean_json = clean_json[:-3]
                
                parsed_segments = json.loads(clean_json.strip())
                st.session_state['radio_segments'] = parsed_segments
                st.session_state['radio_audio'] = manual_audio_path.strip()
                st.success('✅ Segments loaded! You can now proceed to Step 3 (Plan Visuals).')
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f'❌ Invalid JSON: {e}')
        else:
            st.warning('Please provide both segments JSON and audio path.')

# ============================================
# STEP 3 BYPASS - VISUAL PLAN MANUAL INPUT
# ============================================
with st.expander('📋 Step 3 Bypass: Visual Plan - View Prompt & Manual Input', expanded=False):
    if st.session_state.get('radio_segments') and st.session_state.get('parsed_segments'):
        # Generate the same prompt that would be sent to Gemini
        transcript_context = json.dumps(st.session_state['radio_segments'], indent=2)
        clip_list = st.session_state['parsed_segments']
        concise_clips = "\n".join([f"Filename: {c['filename']}\nDescription: {c['description'][:200]}..." for c in clip_list])
        
        visual_plan_prompt = f'''You are an expert video editor. I have a voiceover track.

INPUT DATA:
1. TRANSCRIPT (with timestamps):
{transcript_context}

2. AVAILABLE VISUAL CLIPS (Pre-made files):
{concise_clips}

3. AVAILABLE CHARACTER IMAGES (For explanatory segments with white background + bounce animation):
- "beldam" (BeldamOtherMoterHuman2.webp) - The Beldam / Other Mother in human-like form
- "coraline" (Coraline_Jones_with_one_of_her_outfits.webp) - Coraline Jones character image
- "wybie" (Other_Wybie_waving.webp) - Other Wybie waving
- "beldam_coraline" (TheBeldamCoralineBetterRes.webp) - Beldam with Coraline together

YOUR TASK:
For every segment of the script, assign the BEST, MOST RELEVANT visual from the list that matches the spoken words.

OUTPUT FORMAT: Return a JSON with visual_tool, clip_filename (or character_images + positions), description, and avatar_pose.

RULES:
1. **CHECK FOR TITLE CARDS**: If a segment has "visual_tool": "TITLE_CARD" and "text": "Title Name", YOU MUST KEEP IT AS IS.
2. **VIDEO PLAN INSTRUCTIONS**:
   - **STRATEGY**: Mix "PREMADE_CLIP" with "ANIMATED_IMAGE" for variety. Use "AVATAR_CHAT" *sparingly* (only for "I" statements).
   - **PREMADE_CLIP**: Default choice for narration/analysis.
   - **ANIMATED_IMAGE**: Use when introducing or discussing a SPECIFIC CHARACTER.
   - **AVATAR_CHAT**: Use for personal opinions, theory introductions, asking questions to the audience.
3. **AVATAR POSE**: **ALWAYS REQUIRED**. "neutral", "happy", "smug", "angry", "confused", "skeptical", "surprised", "explaining".

JSON FORMAT:
{{
  "visual_plan": [
    {{
      "segment_id": "txt_0_0",
      "start": 0.0,
      "end": 4.5,
      "text": "Coraline is not just a story...",
      "visual_tool": "PREMADE_CLIP",
      "clip_filename": "Clip_01_Scene_video-Scene-001.mp4",
      "description": "Coraline looking bored...",
      "dynamic_text_overlay": false,
      "avatar_pose": "neutral"
    }}
  ]
}}
'''
        col_prompt1, col_prompt2 = st.columns([3, 1])
        with col_prompt1:
            st.text_area('📋 Prompt to Copy (for Visual Plan):', visual_plan_prompt, height=200, key='visual_plan_prompt_display')
        with col_prompt2:
            st.download_button(
                label="⬇️ Download Prompt",
                data=visual_plan_prompt,
                file_name="visual_plan_prompt.txt",
                mime="text/plain",
                key='download_visual_prompt'
            )
        st.caption('👆 Copy or download this prompt, paste into Gemini Web or another AI, get the JSON response, then upload or paste below.')
    else:
        st.warning('Need to complete Step 2 (Analyze Timing) first to generate the prompt.')
    
    st.markdown('---')
    st.write('**Option 1: Upload JSON File**')
    uploaded_visual_plan = st.file_uploader('Upload Visual Plan JSON:', type=['json'], key='upload_visual_plan')
    
    if uploaded_visual_plan is not None:
        if st.button('✅ Load Uploaded Visual Plan', key='load_uploaded_visual_plan'):
            try:
                content = uploaded_visual_plan.read().decode('utf-8')
                parsed_plan = json.loads(content)
                st.session_state['radio_visual_plan'] = parsed_plan
                st.success('✅ Visual Plan loaded from file! Ready to Render.')
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f'❌ Invalid JSON in file: {e}')
    
    st.write('**Option 2: Paste JSON Text**')
    manual_visual_plan_json = st.text_area('Visual Plan JSON:', height=150, key='manual_visual_plan_input', placeholder='Paste your JSON here...')
    
    if st.button('✅ Use Pasted Visual Plan', key='use_manual_visual_plan'):
        if manual_visual_plan_json.strip():
            try:
                # Clean markdown code blocks if present
                clean_json = manual_visual_plan_json.strip()
                if clean_json.startswith('```json'):
                    clean_json = clean_json[7:]
                if clean_json.startswith('```'):
                    clean_json = clean_json[3:]
                if clean_json.endswith('```'):
                    clean_json = clean_json[:-3]
                
                parsed_plan = json.loads(clean_json.strip())
                st.session_state['radio_visual_plan'] = parsed_plan
                st.success('✅ Visual Plan loaded from manual input! Ready to Render.')
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f'❌ Invalid JSON: {e}')
        else:
            st.warning('Please paste JSON first.')

# ============================================
# STEP 4 NOTE - NO API CALL
# ============================================
with st.expander('🎬 Step 4 Info: Assemble Final Video', expanded=False):
    st.info('ℹ️ **Step 4 does NOT use any API calls.** It uses FFMPEG locally to assemble the video, so no manual bypass is needed.')
    st.write('If Step 4 fails, check:')
    st.write('- FFMPEG is installed and in PATH')
    st.write('- All required video clips exist in the media folder')
    st.write('- Audio file exists at the expected path')

# ============================================
# STEP 5 BYPASS - TEXT OVERLAY MANUAL INPUT
# ============================================
with st.expander('🔤 Step 5 Bypass: Text Overlay - Manual Input', expanded=False):
    st.write('**This bypasses the "Add Text Overlay" API call (Gemini phrase selection).**')
    st.caption('Note: Step 5 still needs Whisper to run locally for word timestamps. Only the Gemini phrase selection can be bypassed.')
    
    st.markdown('---')
    st.write('**📋 Step A: Copy this prompt and paste it into Gemini/ChatGPT:**')
    
    # Build the text overlay prompt dynamically
    original_script_preview = st.session_state.get('radio_script', 'No script loaded yet...')[:3000]
    
    text_overlay_prompt = f'''I have a video essay script and its word-level transcript with timestamps.

ORIGINAL SCRIPT (what I intended to say):
{original_script_preview}

ACTUAL TRANSCRIPT (what Whisper heard - you'll get this after running Step 5 which runs Whisper):
[Whisper will provide this - paste it here or leave as placeholder]

YOUR TASK:
Select 5-10 KEY PHRASES (2-4 words each) that should appear as large, bold "brick-style" text overlays.
These should be:
- Impactful statements ("The door is fake", "She never escaped")
- Key revelations ("The Beldam knew")
- Hook phrases that grab attention

Return a JSON list of objects:
[
    {{
        "phrase": "the door is fake",
        "emphasis": "high"  // "high", "medium", or "low"
    }}
]

RULES:
- Use the EXACT words as they appear in the transcript (matching Whisper output).
- Don't over-use this effect. Pick only the most powerful moments.
- Return ONLY valid JSON, no markdown.
'''
    
    col_prompt1, col_prompt2 = st.columns([3, 1])
    with col_prompt1:
        st.text_area('📋 Text Overlay Prompt:', text_overlay_prompt, height=200, key='step5_prompt_display')
    with col_prompt2:
        st.download_button(
            label="⬇️ Download Prompt",
            data=text_overlay_prompt,
            file_name="step5_text_overlay_prompt.txt",
            mime="text/plain",
            key='download_step5_prompt'
        )
    
    st.markdown('---')
    st.write('**📥 Step B: Paste the AI response (phrase selections) here:**')
    st.caption('Expected format: JSON array of objects with "phrase" and "emphasis" fields')
    manual_phrases_json = st.text_area(
        'Phrase Selections JSON:', 
        height=150, 
        key='manual_phrases_input', 
        placeholder='[\n  {"phrase": "the door is fake", "emphasis": "high"},\n  {"phrase": "she never escaped", "emphasis": "medium"}\n]'
    )
    
    if st.button('✅ Save Phrase Selections for Text Overlay', key='use_manual_phrases'):
        if manual_phrases_json.strip():
            try:
                clean_json = manual_phrases_json.strip()
                if clean_json.startswith('```json'):
                    clean_json = clean_json[7:]
                if clean_json.startswith('```'):
                    clean_json = clean_json[3:]
                if clean_json.endswith('```'):
                    clean_json = clean_json[:-3]
                
                parsed_phrases = json.loads(clean_json.strip())
                st.session_state['manual_text_overlay_phrases'] = parsed_phrases
                st.success(f'✅ {len(parsed_phrases)} phrases saved! Use the Text Overlay button - it will use these instead of calling API.')
            except json.JSONDecodeError as e:
                st.error(f'❌ Invalid JSON: {e}')
        else:
            st.warning('Please paste the phrase selections JSON first.')

if st.session_state['radio_visual_plan']:
    st.subheader('🎬 Visual Plan (Edit Before Render)')
    plan_data = st.session_state['radio_visual_plan']['visual_plan']
    df_plan = pd.DataFrame(plan_data)
    
    # Pre-process: Convert list/dict columns to string for editing
    if 'character_images' in df_plan.columns:
        df_plan['character_images'] = df_plan['character_images'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x) if x is not None else '')

    column_config = {
        'visual_tool': st.column_config.SelectboxColumn('Tool', options=['PREMADE_CLIP', 'ANIMATED_IMAGE', 'TITLE_CARD', 'AVATAR_CHAT', 'MOVIE_CLIP', 'MANIM', 'STOCK', 'IMAGE'], width='medium'),
        'clip_id_from_db': st.column_config.NumberColumn('Clip ID (Legacy)', help='The ID of the movie clip in your markdown file (1-9)', min_value=1, max_value=20, step=1, format='%d'), # kept for backward compat
        'clip_filename': st.column_config.TextColumn('Clip Filename', help='Filename of the pre-made clip'),
        'character_image': st.column_config.SelectboxColumn('Character (Legacy)', options=['beldam', 'coraline', 'wybie', 'beldam_coraline'], help='Single character for ANIMATED_IMAGE (legacy format)'),
        'character_images': st.column_config.TextColumn('Characters (JSON)', help='Array of {image, position} for multiple characters. E.g. [{"image":"beldam","position":"left"}]'),
        'bounce_type': st.column_config.SelectboxColumn('Bounce Type', options=['elastic', 'bounce', 'spring', 'slide_bounce'], help='Animation style for ANIMATED_IMAGE'),
        'seek_time': st.column_config.TextColumn('Seek Time', help='Start time in the source clip (MM:SS)'),
        'avatar_pose': st.column_config.TextColumn('Avatar Pose', help='Expression for the host avatar'),
        'description': st.column_config.TextColumn('Description', width='large'),
        'start': st.column_config.NumberColumn('Start', format='%.2f', disabled=True),
        'end': st.column_config.NumberColumn('End', disabled=True),
        'text': st.column_config.TextColumn('Voiceover Text', disabled=True),
        'segment_id': st.column_config.NumberColumn('ID', disabled=True)
    }
    edited_df = st.data_editor(df_plan, column_order=['text', 'start', 'end', 'visual_tool', 'clip_filename', 'character_images', 'character_image', 'bounce_type', 'clip_id_from_db', 'seek_time', 'avatar_pose', 'description'], column_config=column_config, use_container_width=True, hide_index=True, key='editor_visual_plan')
    if st.button('💾 Confirm Visual Plan Updates'):
        # Post-process: Parse JSON strings back to lists
        if 'character_images' in edited_df.columns:
            def safe_json_load(x):
                try:
                    return json.loads(x) if x.strip() else None
                except:
                    return None
            edited_df['character_images'] = edited_df['character_images'].apply(safe_json_load)
            
        updated_records = edited_df.to_dict('records')
        st.session_state['radio_visual_plan']['visual_plan'] = updated_records
        st.success('✅ Visual Plan Updated! Ready to Render.')
    with st.expander('View Raw JSON'):
        st.json(st.session_state['radio_visual_plan'])

st.markdown('---')
active_scene_data = st.session_state['generated_scene_data'] if st.session_state['generated_scene_data'] else SCENE_DATA

st.subheader('⚡ Automation')
if st.button('✨ Generate ALL Assets (Scripts + Audio + Clips)', help='Generates AI scripts, Audio Voiceovers, and Extracts Movie Clips'):
    if not client:
        st.error('Please provide Gemini API Key in sidebar')
    else:
        with st.status('🚀 Processing Video Assets...', expanded=True) as status:
            curr_progress = st.progress(0)
            total_scenes = len(active_scene_data['scenes'])
            for index, scene in enumerate(active_scene_data['scenes']):
                sid = scene['id']
                st.write(f'**Processing Scene {sid}...**')
                desc = scene.get('description', scene.get('code_or_prompt', 'Scene'))
                current_script = st.session_state['scripts'].get(sid, '')
                if not current_script.strip():
                    st.write('   📝 Writing script...')
                    prompt = f"Write a 1-sentence engaging YouTube-style voiceover script for a video about '{video_topic}'. Focus on this scene: {desc}. Duration: {scene.get('duration', 10)} seconds. Keep it punchy."
                    try:
                        response = client.generate_content(contents=prompt)
                        current_script = response.text.strip()
                        st.session_state['scripts'][sid] = current_script
                        st.session_state[f'script_{sid}'] = current_script
                    except Exception as e:
                        st.error(f"Error generating script for Scene {scene['id']}: {e}")
                else:
                    st.write('   📝 Using existing script...')
                if current_script and (os.path.exists(f'media/vo_{sid}.wav') or os.path.exists(f'media/vo_{sid}.mp3')):
                    st.write('   🔊 Audio already exists. Skipping.')
                else:
                    st.write('   🔊 Generating Audio...')
                    try:
                        generate_voiceover(client, sid, current_script)
                    except Exception as e:
                        st.error(f'Audio failed: {e}')
                if scene['visual_tool'] == 'MOVIE_CLIP':
                    clip_num = scene.get('clip_num', 1)
                    start_time = scene.get('start_time')
                    end_time = scene.get('end_time')
                    video_filename = DEFAULT_CLIP_MAPPING.get(clip_num, '')
                    if video_filename:
                        video_path = os.path.join(movie_folder, video_filename)
                        if os.path.exists(video_path):
                            output_path = f'assets/clip_{sid}.mp4'
                            if os.path.exists(output_path):
                                st.write('   ✂️ Clip already extracted. Skipping.')
                                st.session_state['clip_paths'][sid] = output_path
                            else:
                                st.write(f'   ✂️ Extracting clip ({start_time}-{end_time})...')
                                res = extract_clip_ffmpeg(video_path, start_time, end_time, output_path)
                                if res:
                                    st.session_state['clip_paths'][sid] = res
                        else:
                            st.error(f'   ❌ Source video not found: {video_filename}')
                    else:
                        st.warning(f'   ⚠️ No file mapping for Clip {clip_num}')
                curr_progress.progress((index + 1) / total_scenes)
            status.update(label='✅ All Assets Prepared!', state='complete')
            st.rerun()

tabs = st.tabs([f"Scene {s['id']}" for s in active_scene_data['scenes']] + ['🛠 Final Assembly'])
for i, scene in enumerate(active_scene_data['scenes']):
    with tabs[i]:
        sid = scene['id']
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader(f"Scene {sid}: {scene['visual_tool']}")
            st.info(scene.get('description', 'No description'))
            default_script = scene.get('script', st.session_state['scripts'].get(sid, ''))
            script_text = st.text_area(f'Voiceover Script (S{sid})', value=default_script, key=f'script_{sid}')
            st.session_state['scripts'][sid] = script_text
            if st.button(f'🔊 Preview TTS (S{sid})'):
                if script_text:
                    vo_path = generate_voiceover(client, sid, script_text)
                    st.audio(vo_path)
                else:
                    st.warning('Please enter a script first')
        with col2:
            st.subheader('Visual Asset')
            if scene['visual_tool'] == 'MOVIE_CLIP':
                clip_num = scene.get('clip_num', 1)
                start_time = scene.get('start_time', '00:00')
                end_time = scene.get('end_time', '00:30')
                st.info(f'🎬 Clip {clip_num}: {start_time} → {end_time}')
                video_filename = DEFAULT_CLIP_MAPPING.get(clip_num, '')
                video_path = os.path.join(movie_folder, video_filename) if video_filename else ''
                if video_filename:
                    st.caption(f'📁 Source: {video_filename}')
                    if os.path.exists(video_path):
                        if st.button(f'✂️ Auto-Extract Clip (S{sid})', key=f'extract_{sid}'):
                            with st.spinner(f'Cutting clip from {start_time} to {end_time}...'):
                                output_path = f'assets/clip_{sid}.mp4'
                                result = extract_clip_ffmpeg(video_path, start_time, end_time, output_path)
                                if result:
                                    st.session_state['clip_paths'][sid] = result
                                    st.success('✅ Clip extracted!')
                                    st.video(result)
                                else:
                                    st.error('❌ FFMPEG extraction failed. Is FFMPEG installed?')
                        existing_clip = st.session_state['clip_paths'].get(sid)
                        if existing_clip and os.path.exists(existing_clip):
                            st.video(existing_clip)
                    else:
                        st.error(f'Video file not found: {video_path}')
                else:
                    st.error(f'No mapping for Clip {clip_num}')
            elif scene['visual_tool'] == 'EXTERNAL_MOVIE_CLIP':
                st.warning(f"Required: {scene.get('code_or_prompt', 'Upload clip')}")
                uploaded_file = st.file_uploader(f'Upload Clip for S{sid}', type=['mp4', 'mov'], key=f'upload_{sid}')
                if uploaded_file:
                    path = f'assets/scene_{sid}.mp4'
                    with open(path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state['clip_paths'][sid] = path
                    st.success('File Uploaded')
            elif scene['visual_tool'] == 'MANIM':
                code = st.text_area('Manim Code', value=scene.get('code_or_prompt', ''), height=200, key=f'code_{sid}')
                if st.button(f'🔨 Render Manim (S{sid})', key=f'manim_{sid}'):
                    with st.spinner('Rendering Manim...'):
                        res = render_manim(sid, code)
                        if res:
                            st.session_state['clip_paths'][sid] = res
                            st.video(res)
            elif scene['visual_tool'] == 'STOCK_VIDEO':
                query = st.text_input('Stock Query', value=scene.get('code_or_prompt', ''), key=f'q_{sid}')
                if st.button(f'🔍 Fetch Pexels (S{sid})', key=f'pexels_{sid}'):
                    with st.spinner('Searching Pexels...'):
                        res = fetch_pexels_video(query, sid)
                        if res:
                            st.session_state['clip_paths'][sid] = res
                            st.video(res)
            elif scene['visual_tool'] == 'THREEJS':
                js_code = st.text_area('p5.js Code', value=scene['code_or_prompt'], height=200, key=f'js_{sid}')
                if st.button(f'🌐 Render JS (S{sid})'):
                    with st.spinner('Rendering Creative Code via Selenium...'):
                        res = render_threejs(sid, js_code, scene['duration'])
                        if res:
                            st.session_state['clip_paths'][sid] = res
                            st.video(res)
            elif scene['visual_tool'] == 'AI_IMAGE':
                st.info(f"Prompt: {scene['code_or_prompt']}")
                uploaded_img = st.file_uploader(f'Upload Generated Image (S{sid})', type=['png', 'jpg', 'jpeg'])
                if uploaded_img:
                    img_path = f'assets/scene_{sid}.png'
                    with open(img_path, 'wb') as f:
                        f.write(uploaded_img.getbuffer())
                    img_clip = ImageClip(img_path).set_duration(scene['duration'])
                    vid_path = f'media/img_clip_{sid}.mp4'
                    img_clip.write_videofile(vid_path, fps=24)
                    st.session_state['clip_paths'][sid] = vid_path
                    st.image(img_path)

with tabs[-1]:
    st.header('Assemble Final Masterpiece')
    available_clips = [sid for sid in st.session_state['clip_paths'] if st.session_state['clip_paths'][sid]]
    st.info(f"📊 Available clips: {len(available_clips)} / {len(active_scene_data['scenes'])} scenes")
    if st.button('🎬 Assemble Full Video (FFMPEG)'):
        st.write('🚀 Starting FFMPEG Assembly...')
        os.makedirs('build', exist_ok=True)
        files_to_concat = []
        prog = st.progress(0)
        total = len(active_scene_data['scenes'])
        try:
            for i, scene in enumerate(active_scene_data['scenes']):
                sid = scene['id']
                if sid not in st.session_state['clip_paths']:
                    st.warning(f'⚠️ Skipping Scene {sid}: No video clip found.')
                    continue
                video_path = st.session_state['clip_paths'][sid]
                audio_path = f'media/vo_{sid}.wav'
                if not os.path.exists(audio_path):
                    audio_path = f'media/vo_{sid}.mp3'
                if not os.path.exists(audio_path):
                    script = st.session_state['scripts'].get(sid, '')
                    if script:
                        st.info(f'🎤 Generating missing audio for Scene {sid}...')
                        audio_path = generate_voiceover(client, sid, script)
                    else:
                        st.warning(f'⚠️ Skipping Scene {sid}: No audio/script.')
                        continue
                output_part = f'build/part_{sid}.mp4'
                
                # SUPPORT FOR NEW VISUAL TOOLS
                if scene['visual_tool'] == 'KENBURNS_IMAGE':
                    st.write(f'   🖼️ Rendering Ken Burns for Scene {sid}...')
                    # Find image path - try matching description or just use a default placeholder
                    img_path = 'assets/placeholder.png' # You might want to pick from search
                    video_path = render_kenburns_image(sid, img_path, duration=scene['duration'])
                elif scene['visual_tool'] == 'AVATAR_EXPLANATION':
                    st.write(f'   🎤 Rendering speaking avatar for Scene {sid}...')
                    video_path = render_avatar_explanation(sid, scene.get('avatar_pose', 'neutral'), duration=scene['duration'])
                
                v_dur = get_file_duration(video_path)
                a_dur = get_file_duration(audio_path)
                st.write(f'   ⏱️ Scene {sid}: Video={v_dur:.2f}s, Audio={a_dur:.2f}s')
                input_args = ['-i', video_path, '-i', audio_path]
                video_filters = 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1'
                if v_dur > 0 and a_dur > 0 and (v_dur < a_dur):
                    stretch_factor = a_dur / v_dur
                    st.write(f'   🐢 Stretching video by {stretch_factor:.2f}x to fit audio')
                    video_filters = video_filters + f',setpts={stretch_factor}*PTS'
                else:
                    video_filters = video_filters + ',setpts=PTS-STARTPTS'
                video_filters = video_filters + ',fps=30,format=yuv420p'
                cmd = ['ffmpeg', '-y'] + input_args + ['-filter_complex', f'[0:v]{video_filters}[v];[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a]', '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-c:a', 'aac', '-shortest', output_part]
                st.write(f'⚙️ Rendering Scene {sid}...')
                process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if process.returncode != 0:
                    st.error(f'❌ FFMPEG Error Scene {sid}: {process.stderr.decode()}')
                    continue
                if os.path.exists(output_part):
                    abs_path = os.path.abspath(output_part).replace('\\', '/')
                    files_to_concat.append(f"file '{abs_path}'")
                prog.progress((i + 1) / total)
            if files_to_concat:
                concat_file = 'build/concat_list.txt'
                with open(concat_file, 'w') as f:
                    f.write('\n'.join(files_to_concat))
                final_path = 'media/final_output.mp4'
                st.write('🔗 Stitching final video...')
                concat_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file, '-c', 'copy', final_path]
                subprocess.run(concat_cmd, check=True)
                st.success('✅ Video Assembled Successfully!')
                st.video(final_path)
                with open(final_path, 'rb') as f:
                    st.download_button('Download Full Video', f, file_name='coraline_analysis.mp4')
            else:
                st.error('No valid scenes to assemble.')
        except Exception as e:
            st.error(f'Assembly Critical Error: {e}')

st.markdown('---')
st.caption('Expert AI Video Creator Tool | Manim | MoviePy | Gemini 2.0')
