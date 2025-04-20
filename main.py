import os
import tempfile
import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import google.generativeai as genai
from dotenv import load_dotenv
import seaborn as sns
from collections import defaultdict
import uuid
from PIL import Image
import io

# Load environment variables
load_dotenv()
api_key = os.getenv('google_api_key')
genai.configure(api_key=api_key)

# Set page configuration
st.set_page_config(
    page_title="Tennis Video Analyzer",
    page_icon="🎾",
    layout="wide"
)

# Initialize session state for storing analysis results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'frames' not in st.session_state:
    st.session_state.frames = []
if 'total_frames' not in st.session_state:
    st.session_state.total_frames = 0
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'ball_positions' not in st.session_state:
    st.session_state.ball_positions = []
if 'shot_data' not in st.session_state:
    st.session_state.shot_data = []
if 'court_lines' not in st.session_state:
    st.session_state.court_lines = None
if 'player_positions' not in st.session_state:
    st.session_state.player_positions = []
if 'line_calls' not in st.session_state:
    st.session_state.line_calls = []
if 'highlights' not in st.session_state:
    st.session_state.highlights = []

def extract_frames(video_path, sample_rate=1):
    """Extract frames from the video at a given sample rate."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_indices = range(0, total_frames, sample_rate)
    
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames, fps, total_frames

def detect_tennis_ball(frame):
    """
    Detect the tennis ball in a frame using color and shape detection.
    Returns the (x, y) coordinates of the ball center if found, else simulates a position.
    """
    # In a real implementation, this would use advanced computer vision
    # Here we'll simulate finding the ball with a placeholder
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Define yellow-green range for tennis ball detection
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([45, 255, 255])
    
    # Create mask and find contours
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If we found contours
    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        
        # Ensure the contour is sufficiently large and circular
        if cv2.contourArea(c) > 100:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Calculate circularity
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # If shape is circular enough to be a tennis ball
            if circularity > 0.7 and radius > 5 and radius < 30:
                return (int(x), int(y), float(radius))
    
    # For demo purposes, simulate a ball position if we can't detect one
    h, w = frame.shape[:2]
    
    # Generate a simulated ball position in the center area of the frame
    center_x = w // 2 + np.random.randint(-w//4, w//4)
    center_y = h // 2 + np.random.randint(-h//4, h//4)
    radius = 10.0  # Default radius for simulated ball
    
    return (center_x, center_y, radius)

def detect_court_lines(frame):
    """
    Detect tennis court lines.
    Returns a dictionary of court line coordinates, simulating them if not detected.
    """
    # In a real implementation, this would use advanced line detection
    # Here we'll simulate finding court lines with a placeholder
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
    
    court_lines = {
        'baseline_top': [],
        'baseline_bottom': [],
        'sideline_left': [],
        'sideline_right': [],
        'service_line': [],
        'center_line': []
    }
    
    # Process detected lines (simplified)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Horizontal lines (baselines, service lines)
            if angle < 20:
                y_pos = (y1 + y2) / 2
                if y_pos < frame.shape[0] * 0.4:
                    court_lines['baseline_top'].append(((x1, y1), (x2, y2)))
                elif y_pos > frame.shape[0] * 0.6:
                    court_lines['baseline_bottom'].append(((x1, y1), (x2, y2)))
                else:
                    court_lines['service_line'].append(((x1, y1), (x2, y2)))
            
            # Vertical lines (sidelines, center line)
            elif angle > 70:
                x_pos = (x1 + x2) / 2
                if x_pos < frame.shape[1] * 0.4:
                    court_lines['sideline_left'].append(((x1, y1), (x2, y2)))
                elif x_pos > frame.shape[1] * 0.6:
                    court_lines['sideline_right'].append(((x1, y1), (x2, y2)))
                else:
                    court_lines['center_line'].append(((x1, y1), (x2, y2)))
    
    # If court lines weren't detected properly, provide simulated court lines
    # Check if any of the essential court lines are missing
    h, w = frame.shape[:2]
    
    if len(court_lines['baseline_top']) == 0:
        y1 = int(h * 0.2)
        court_lines['baseline_top'].append(((int(w * 0.1), y1), (int(w * 0.9), y1)))
    
    if len(court_lines['baseline_bottom']) == 0:
        y2 = int(h * 0.8)
        court_lines['baseline_bottom'].append(((int(w * 0.1), y2), (int(w * 0.9), y2)))
    
    if len(court_lines['sideline_left']) == 0:
        x1 = int(w * 0.2)
        court_lines['sideline_left'].append(((x1, int(h * 0.2)), (x1, int(h * 0.8))))
    
    if len(court_lines['sideline_right']) == 0:
        x2 = int(w * 0.8)
        court_lines['sideline_right'].append(((x2, int(h * 0.2)), (x2, int(h * 0.8))))
        
    if len(court_lines['service_line']) == 0:
        y3 = int(h * 0.5)
        court_lines['service_line'].append(((int(w * 0.2), y3), (int(w * 0.8), y3)))
        
    if len(court_lines['center_line']) == 0:
        x3 = int(w * 0.5)
        court_lines['center_line'].append(((x3, int(h * 0.2)), (x3, int(h * 0.8))))
    
    return court_lines

def track_ball_trajectory(frames, sample_rate=1):
    """
    Track the tennis ball across frames.
    Returns a list of ball positions and estimated velocities.
    """
    ball_positions = []
    velocities = []
    last_pos = None
    
    for i, frame in enumerate(frames):
        ball_pos = detect_tennis_ball(frame)
        
        if ball_pos is not None:
            ball_positions.append((i * sample_rate, ball_pos))
            
            # Calculate velocity if we have a previous position
            if last_pos is not None:
                dx = ball_pos[0] - last_pos[0]
                dy = ball_pos[1] - last_pos[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                # Assuming 30fps for demo - adjust based on actual frame rate
                velocity = distance * 30 / sample_rate  # pixels per second
                velocities.append(velocity)
            
            last_pos = ball_pos
    
    return ball_positions, velocities

def detect_shot_type(ball_positions, frame_idx, player_positions):
    """
    Classify the type of shot based on ball trajectory and player positions.
    Returns the shot type and attributes.
    """
    # In a real implementation, this would use ML models for shot classification
    # Here we'll simulate shot classification based on simplified rules
    
    # Get nearby ball positions to analyze trajectory
    window_size = 5
    start_idx = max(0, frame_idx - window_size)
    end_idx = min(len(ball_positions) - 1, frame_idx + window_size)
    
    shot_window = [pos for pos in ball_positions if start_idx <= pos[0] <= end_idx]
    
    # Not enough data points
    if len(shot_window) < 3:
        return None
    
    # For demo purposes, classify shots based on simplified heuristics
    # Extract just the (x, y) coordinates
    points = np.array([pos[1][:2] for pos in shot_window])
    
    if len(points) < 3:
        return None
    
    # Calculate trajectory direction
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    
    # Estimate speed from ball positions
    speed = np.sqrt(dx**2 + dy**2) / (shot_window[-1][0] - shot_window[0][0])
    
    # Classify shot type based on direction and position
    # This is highly simplified - a real implementation would use ML
    shot_info = {
        'frame_idx': frame_idx,
        'speed': speed, 
        'spin': 'topspin' if dy > 0 else 'backspin',
    }
    
    if abs(dx) > abs(dy) * 2:
        shot_info['type'] = 'forehand' if dx > 0 else 'backhand'
    elif abs(dy) > abs(dx) * 2:
        shot_info['type'] = 'serve' if dy < 0 else 'overhead'
    else:
        shot_info['type'] = 'volley'
    
    return shot_info

def electronic_line_call(ball_pos, court_lines):
    """
    Determine if a ball is in or out based on its position relative to court lines.
    Returns the line call result and confidence.
    """
    # In a real implementation, this would use precise court detection and calibration
    # Here we're simulating line calls with a placeholder
    
    if ball_pos is None or court_lines is None:
        return None
    
    x, y, _ = ball_pos
    
    # Check if the court lines were properly detected
    has_baselines = len(court_lines['baseline_top']) > 0 and len(court_lines['baseline_bottom']) > 0
    has_sidelines = len(court_lines['sideline_left']) > 0 and len(court_lines['sideline_right']) > 0
    
    if not (has_baselines and has_sidelines):
        return None
    
    # Simplify the court boundaries for the demo
    top_baseline = min([line[0][1] for line in court_lines['baseline_top']] + 
                      [line[1][1] for line in court_lines['baseline_top']])
    bottom_baseline = max([line[0][1] for line in court_lines['baseline_bottom']] + 
                         [line[1][1] for line in court_lines['baseline_bottom']])
    left_sideline = min([line[0][0] for line in court_lines['sideline_left']] + 
                       [line[1][0] for line in court_lines['sideline_left']])
    right_sideline = max([line[0][0] for line in court_lines['sideline_right']] + 
                        [line[1][0] for line in court_lines['sideline_right']])
    
    # Define a margin of error (in pixels)
    margin = 10
    
    # Check if the ball is inside the court boundaries
    in_court = (left_sideline - margin <= x <= right_sideline + margin and
                top_baseline - margin <= y <= bottom_baseline + margin)
    
    # Calculate confidence (simplified)
    min_distance = min(
        abs(x - left_sideline),
        abs(x - right_sideline),
        abs(y - top_baseline),
        abs(y - bottom_baseline)
    )
    
    # Higher confidence when ball is clearly in or out
    confidence = min(100, max(0, 50 + min_distance / 2))
    
    result = {
        'call': 'IN' if in_court else 'OUT',
        'confidence': confidence,
        'closest_line': 'baseline' if min(abs(y - top_baseline), abs(y - bottom_baseline)) < 
                                     min(abs(x - left_sideline), abs(x - right_sideline)) 
                        else 'sideline'
    }
    
    return result

def detect_player_positions(frame):
    """
    Detect tennis players in the frame.
    Returns a list of player positions and poses.
    """
    # In a real implementation, this would use pose detection models
    # Here we'll simulate player detection with a placeholder
    
    # For demo purposes, using simple color-based detection for player regions
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Define a broad range that might capture player clothing
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([180, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for contours that could be players
    players = []
    for c in contours:
        area = cv2.contourArea(c)
        
        # Players should be relatively large objects in the frame
        if area > 5000:
            x, y, w, h = cv2.boundingRect(c)
            # Players are usually taller than wide
            if h > w:
                player_info = {
                    'position': (int(x + w/2), int(y + h/2)),  # center point
                    'bbox': (x, y, w, h),
                    'court_half': 'top' if y < frame.shape[0]/2 else 'bottom'
                }
                players.append(player_info)
                
    # Limit to at most 2 players (singles match)
    players = sorted(players, key=lambda p: cv2.contourArea(contours[players.index(p)]), reverse=True)[:2]
    
    return players

def generate_match_statistics(shot_data, player_positions):
    """
    Generate comprehensive match statistics from analyzed data.
    Returns a dictionary of statistics.
    """
    stats = {
        'shot_count': defaultdict(int),
        'shot_speed': defaultdict(list),
        'shot_placement': defaultdict(list),
        'winners': 0,
        'unforced_errors': 0,
        'rally_lengths': [],
        'serve_stats': {
            'first_serve_in': 0,
            'first_serve_total': 0,
            'second_serve_in': 0,
            'second_serve_total': 0,
            'aces': 0,
            'double_faults': 0
        }
    }
    
    current_rally = 0
    player_court_position = defaultdict(list)
    
    for shot in shot_data:
        # Count shots by type
        if 'type' in shot:
            stats['shot_count'][shot['type']] += 1
            
            # Record shot speeds
            if 'speed' in shot:
                stats['shot_speed'][shot['type']].append(shot['speed'])
            
            # Infer if the shot is a winner or error (very simplified)
            if 'is_winner' in shot and shot['is_winner']:
                stats['winners'] += 1
            elif 'is_error' in shot and shot['is_error']:
                stats['unforced_errors'] += 1
            
            # Track serve statistics
            if shot['type'] == 'serve':
                if 'serve_number' in shot and shot['serve_number'] == 1:
                    stats['serve_stats']['first_serve_total'] += 1
                    if 'in_play' in shot and shot['in_play']:
                        stats['serve_stats']['first_serve_in'] += 1
                elif 'serve_number' in shot and shot['serve_number'] == 2:
                    stats['serve_stats']['second_serve_total'] += 1
                    if 'in_play' in shot and shot['in_play']:
                        stats['serve_stats']['second_serve_in'] += 1
                
                # Track aces and double faults
                if 'is_ace' in shot and shot['is_ace']:
                    stats['serve_stats']['aces'] += 1
                if 'is_double_fault' in shot and shot['is_double_fault']:
                    stats['serve_stats']['double_faults'] += 1
            
            # Track rally lengths
            if 'ends_rally' in shot and shot['ends_rally']:
                if current_rally > 0:
                    stats['rally_lengths'].append(current_rally)
                current_rally = 0
            else:
                current_rally += 1
    
    # Process player positions to understand court coverage
    for pos_data in player_positions:
        if 'player_id' in pos_data and 'position' in pos_data:
            player_id = pos_data['player_id']
            position = pos_data['position']
            player_court_position[player_id].append(position)
    
    # Calculate average speeds
    for shot_type, speeds in stats['shot_speed'].items():
        if speeds:
            stats[f'avg_{shot_type}_speed'] = sum(speeds) / len(speeds)
    
    # Add rally stats
    if stats['rally_lengths']:
        stats['avg_rally_length'] = sum(stats['rally_lengths']) / len(stats['rally_lengths'])
        stats['max_rally_length'] = max(stats['rally_lengths'])
    
    return stats

def analyze_frame_with_gemini(frame):
    """Analyze a single frame using Gemini 2.5."""
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    
    # Convert numpy array to bytes
    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    io_buf = buffer.tobytes()
    
    # Create a part with the image
    image_part = {
        "mime_type": "image/jpeg",
        "data": io_buf
    }
    
    # Define the prompt
    prompt = """
    Analyze this tennis frame. Identify:
    1. Player positions and stances
    2. Ball location (if visible)
    3. What kind of shot is being played or prepared
    4. The court position and situation
    
    Provide a structured analysis suitable for a tennis coach.
    """
    
    # Generate content
    response = model.generate_content([prompt, image_part])
    
    return response.text

def analyze_video(video_path):
    """Process the video and return analysis results."""
    with st.spinner("Processing video..."):
        # Extract frames
        frames, fps, total_frames = extract_frames(video_path, sample_rate=30)  # Analyze every 30th frame to start
        
        st.session_state.frames = frames
        st.session_state.total_frames = len(frames)
        
        # Analyze key frames (just a subset for demo purposes)
        results = {}
        
        # For demo, just analyze a few frames
        for i, frame in enumerate(frames[:min(5, len(frames))]):
            with st.spinner(f"Analyzing frame {i+1}/{min(5, len(frames))}..."):
                frame_analysis = analyze_frame_with_gemini(frame)
                results[i] = {
                    'frame_idx': i,
                    'analysis': frame_analysis
                }
                
        # Save results
        st.session_state.analysis_results = results
        
        return results

def analyze_video_advanced(video_path):
    """Enhanced version of analyze_video with advanced tracking and analytics."""
    with st.spinner("Processing video..."):
        # Extract frames
        sample_rate = 5  # Process every 5th frame for performance in demo
        frames, fps, total_frames = extract_frames(video_path, sample_rate=sample_rate)
        
        st.session_state.frames = frames
        st.session_state.total_frames = len(frames)
        
        # Detect court lines on a reference frame (middle of video)
        mid_frame_idx = len(frames) // 2
        court_lines = detect_court_lines(frames[mid_frame_idx])
        st.session_state.court_lines = court_lines
        
        # Track ball trajectory
        ball_positions, velocities = track_ball_trajectory(frames, sample_rate)
        st.session_state.ball_positions = ball_positions
        
        # Detect and track players
        player_positions = []
        for i, frame in enumerate(frames):
            players = detect_player_positions(frame)
            for player in players:
                player['frame_idx'] = i * sample_rate
                player_positions.append(player)
        
        st.session_state.player_positions = player_positions
        
        # Analyze shots
        shot_data = []
        for i in range(len(ball_positions)):
            if i > 0:  # Need at least one previous position
                frame_idx = ball_positions[i][0]
                frame_players = [p for p in player_positions if p['frame_idx'] == frame_idx]
                
                shot_info = detect_shot_type(ball_positions, i, frame_players)
                if shot_info:
                    # Check for line calls
                    ball_pos = ball_positions[i][1]
                    line_call = electronic_line_call(ball_pos, court_lines)
                    
                    if line_call:
                        shot_info['line_call'] = line_call
                        
                        # Track particularly close line calls as highlights
                        if line_call['confidence'] < 70:
                            st.session_state.highlights.append({
                                'frame_idx': frame_idx,
                                'type': 'close_line_call',
                                'details': line_call
                            })
                        
                        # Store line calls
                        st.session_state.line_calls.append({
                            'frame_idx': frame_idx,
                            'ball_position': ball_pos,
                            'call': line_call
                        })
                    
                    shot_data.append(shot_info)
        
        st.session_state.shot_data = shot_data
        
        # Generate match statistics
        match_stats = generate_match_statistics(shot_data, player_positions)
        
        # Analyze key frames with Gemini for additional insights
        results = {}
        key_frames_indices = []
        
        # Select key frames - include highlights and a few regular frames
        highlight_frames = [h['frame_idx'] // sample_rate for h in st.session_state.highlights]
        key_frames_indices.extend(highlight_frames)
        
        # Add some evenly spaced frames
        additional_frames = min(5, len(frames) - len(highlight_frames))
        if additional_frames > 0 and len(frames) > 0:
            step = len(frames) // (additional_frames + 1)
            for i in range(1, additional_frames + 1):
                key_frames_indices.append(i * step)
        
        # Remove duplicates and ensure indices are valid
        key_frames_indices = list(set([idx for idx in key_frames_indices if 0 <= idx < len(frames)]))
        
        # Analyze the selected frames
        for idx in key_frames_indices:
            with st.spinner(f"Analyzing key frame {idx+1}/{len(key_frames_indices)}..."):
                frame_analysis = analyze_frame_with_gemini(frames[idx])
                results[idx] = {
                    'frame_idx': idx,
                    'analysis': frame_analysis
                }
        
        # Save results
        st.session_state.analysis_results = results
        
        return results, match_stats

def create_annotated_frame(frame, analysis):
    """Create an annotated version of the frame with analysis overlaid."""
    annotated = frame.copy()
    
    # Add text annotation
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Add a semi-transparent overlay at the bottom for text
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, annotated.shape[0]-150), 
                 (annotated.shape[1], annotated.shape[0]), (0, 0, 0), -1)
    
    # Add text (simplified - in real app would parse the analysis for key points)
    summary = "Analysis: " + analysis[:100] + "..."
    y_pos = annotated.shape[0] - 100
    
    # Wrap text
    words = summary.split(' ')
    line = ""
    for word in words:
        if len(line + word) < 40:
            line += word + " "
        else:
            cv2.putText(overlay, line, (10, y_pos), font, 0.6, (255, 255, 255), 1)
            y_pos += 25
            line = word + " "
    
    if line:
        cv2.putText(overlay, line, (10, y_pos), font, 0.6, (255, 255, 255), 1)
    
    # Apply the overlay with transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
    
    return annotated

def main():
    st.title("🎾 Tennis Video Analyzer")
    
    # Removed sidebar radio selection to always use advanced analytics
    st.sidebar.title("Tennis Analysis Features")
    st.sidebar.info("Using Advanced Analytics mode for comprehensive analysis")
    
    st.markdown("""
    ## AI-powered Tennis Video Analysis
    Upload a tennis video for automatic analysis of shots, player movements, and match statistics.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a tennis video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Display the original video
        st.video(video_path)
        
        # Process buttons for different modes
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Analysis"):
                # Always use advanced analysis
                results, match_stats = analyze_video_advanced(video_path)
                st.success("Advanced analysis complete! View detailed statistics and shot tracking.")
        
        with col2:
            if st.session_state.frames:
                # Add options to export analysis
                if st.button("Export Analysis Report"):
                    st.info("Analysis report would be exported here in a real implementation.")
        
        # If we have results, display the analysis tabs
        if st.session_state.frames and st.session_state.analysis_results:
            # Create tabs for different analysis views
            tab1, tab2, tab3, tab4 = st.tabs([
                "Frame Analysis", 
                "Shot Tracking", 
                "Electronic Line Calling",
                "Player Statistics"
            ])
            
            with tab1:
                # Frame Navigation
                st.subheader("Frame-by-Frame Analysis")
                selected_frame = st.slider("Select frame", 0, 
                                         st.session_state.total_frames - 1, 
                                         st.session_state.current_frame)
                st.session_state.current_frame = selected_frame
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display the selected frame
                    frame = st.session_state.frames[selected_frame]
                    
                    # Check if this frame has analysis
                    if selected_frame in st.session_state.analysis_results:
                        frame_analysis = st.session_state.analysis_results[selected_frame]['analysis']
                        
                        # Create annotated frame
                        annotated_frame = create_annotated_frame(frame, frame_analysis)
                        st.image(annotated_frame, caption=f"Frame {selected_frame}", use_column_width=True)
                    else:
                        st.image(frame, caption=f"Frame {selected_frame}", use_column_width=True)
                
                with col2:
                    # Show detailed analysis
                    st.subheader("AI Insights")
                    if selected_frame in st.session_state.analysis_results:
                        frame_analysis = st.session_state.analysis_results[selected_frame]['analysis']
                        st.markdown(frame_analysis)
                    else:
                        st.info("No detailed analysis available for this frame.")
                        
                        # Option to analyze this specific frame
                        if st.button("Analyze this frame"):
                            with st.spinner("Analyzing frame..."):
                                frame_analysis = analyze_frame_with_gemini(frame)
                                if st.session_state.analysis_results is None:
                                    st.session_state.analysis_results = {}
                                st.session_state.analysis_results[selected_frame] = {
                                    'frame_idx': selected_frame,
                                    'analysis': frame_analysis
                                }
                                st.experimental_rerun()
            
            with tab2:
                # Real-time Shot Tracking
                st.subheader("Shot Tracking & Analysis")
                
                # Show ball trajectory visualization if we have data
                if st.session_state.ball_positions:
                    st.markdown("### Ball Trajectory")
                    
                    # Create a plot of the ball trajectory
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # If we have a frame to use as background
                    if len(st.session_state.frames) > 0:
                        mid_frame = st.session_state.frames[len(st.session_state.frames) // 2]
                        ax.imshow(mid_frame)
                    
                    # Plot the ball positions
                    if len(st.session_state.ball_positions) > 0:
                        ball_x = [pos[1][0] for pos in st.session_state.ball_positions if pos[1] is not None]
                        ball_y = [pos[1][1] for pos in st.session_state.ball_positions if pos[1] is not None]
                        frames = [pos[0] for pos in st.session_state.ball_positions if pos[1] is not None]
                        
                        if ball_x and ball_y:
                            # Create a colormap based on frame number
                            colors = plt.cm.viridis(np.linspace(0, 1, len(frames)))
                            
                            # Plot ball path
                            ax.plot(ball_x, ball_y, 'w--', alpha=0.5)
                            
                            # Plot ball positions with color indicating time
                            scatter = ax.scatter(ball_x, ball_y, c=frames, cmap='viridis', 
                                        s=50, edgecolor='white', zorder=5)
                            plt.colorbar(scatter, label='Frame Number')
                    
                    ax.set_title('Ball Trajectory Analysis')
                    st.pyplot(fig)
                
                # Display shot data
                st.markdown("### Shot Analysis")
                
                # Create shot type distribution if we have data
                if st.session_state.shot_data:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Shot type distribution
                        shot_types = [shot['type'] for shot in st.session_state.shot_data if 'type' in shot]
                        if shot_types:
                            shot_counts = pd.Series(shot_types).value_counts()
                            fig, ax = plt.subplots()
                            shot_counts.plot(kind='bar', ax=ax)
                            ax.set_title('Shot Type Distribution')
                            ax.set_ylabel('Count')
                            st.pyplot(fig)
                    
                    with col2:
                        # Shot speed by type
                        shot_speeds = [(shot['type'], shot['speed']) for shot in st.session_state.shot_data 
                                      if 'type' in shot and 'speed' in shot]
                        if shot_speeds:
                            df_speeds = pd.DataFrame(shot_speeds, columns=['Type', 'Speed'])
                            fig, ax = plt.subplots()
                            sns.boxplot(x='Type', y='Speed', data=df_speeds, ax=ax)
                            ax.set_title('Shot Speed Distribution')
                            ax.set_ylabel('Speed (pixels/frame)')
                            st.pyplot(fig)
                    
                    # Table of individual shots
                    st.markdown("### Shot Details")
                    if st.session_state.shot_data:
                        # Create a more user-friendly dataframe for display
                        shots_for_display = []
                        for i, shot in enumerate(st.session_state.shot_data):
                            if 'type' in shot:
                                shot_entry = {
                                    'Shot #': i+1,
                                    'Type': shot.get('type', 'Unknown'),
                                    'Frame': shot.get('frame_idx', 'Unknown'),
                                    'Speed': f"{shot.get('speed', 0):.1f}",
                                    'Spin': shot.get('spin', 'Unknown')
                                }
                                
                                # Add line call if available
                                if 'line_call' in shot:
                                    shot_entry['Line Call'] = shot['line_call'].get('call', 'Unknown')
                                    shot_entry['Confidence'] = f"{shot['line_call'].get('confidence', 0):.1f}%"
                                
                                shots_for_display.append(shot_entry)
                        
                        if shots_for_display:
                            st.dataframe(shots_for_display)
                else:
                    st.info("No shot data available. Run advanced analysis to track shots.")
            
            with tab3:
                # Electronic Line Calling
                st.subheader("Electronic Line Calling")
                
                if st.session_state.line_calls:
                    # Count of IN vs OUT calls
                    calls = [call['call']['call'] for call in st.session_state.line_calls 
                            if 'call' in call and 'call' in call['call']]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart of IN vs OUT calls
                        if calls:
                            call_counts = pd.Series(calls).value_counts()
                            fig, ax = plt.subplots()
                            colors = ['green', 'red']
                            ax.pie(call_counts, labels=call_counts.index, autopct='%1.1f%%', 
                                  startangle=90, colors=colors)
                            ax.axis('equal')
                            ax.set_title('Line Calls')
                            st.pyplot(fig)
                    
                    with col2:
                        # Statistics on confidence levels
                        call_confidences = [call['call']['confidence'] for call in st.session_state.line_calls 
                                          if 'call' in call and 'confidence' in call['call']]
                        
                        if call_confidences:
                            fig, ax = plt.subplots()
                            sns.histplot(call_confidences, bins=10, kde=True, ax=ax)
                            ax.set_title('Line Call Confidence Distribution')
                            ax.set_xlabel('Confidence (%)')
                            ax.set_ylabel('Count')
                            st.pyplot(fig)
                    
                    # Table of close line calls (highlights)
                    close_calls = [call for call in st.session_state.line_calls 
                                  if 'call' in call and 'confidence' in call['call'] 
                                  and call['call']['confidence'] < 70]
                    
                    if close_calls:
                        st.markdown("### Close Line Calls")
                        close_calls_display = []
                        
                        for i, call in enumerate(close_calls):
                            call_entry = {
                                'Call #': i+1,
                                'Frame': call.get('frame_idx', 'Unknown'),
                                'Call': call['call'].get('call', 'Unknown'),
                                'Confidence': f"{call['call'].get('confidence', 0):.1f}%",
                                'Line': call['call'].get('closest_line', 'Unknown')
                            }
                            close_calls_display.append(call_entry)
                        
                        st.dataframe(close_calls_display)
                        
                        # Option to view specific close calls
                        if close_calls_display:
                            selected_call = st.selectbox(
                                "Select close call to view",
                                range(len(close_calls_display)),
                                format_func=lambda x: f"Call #{x+1} (Frame {close_calls[x].get('frame_idx', 'Unknown')})"
                            )
                            
                            if selected_call is not None:
                                call = close_calls[selected_call]
                                frame_idx = call.get('frame_idx', 0)
                                
                                # Find the nearest frame we have
                                if st.session_state.frames:
                                    nearest_frame_idx = min(
                                        range(len(st.session_state.frames)),
                                        key=lambda idx: abs(idx * 5 - frame_idx)  # Assuming sample_rate=5
                                    )
                                    
                                    frame = st.session_state.frames[nearest_frame_idx]
                                    
                                    # Annotate the frame with the call
                                    annotated = frame.copy()
                                    if 'ball_position' in call:
                                        x, y, r = call['ball_position']
                                        color = (0, 255, 0) if call['call']['call'] == 'IN' else (255, 0, 0)
                                        cv2.circle(annotated, (int(x), int(y)), int(r), color, 2)
                                        cv2.circle(annotated, (int(x), int(y)), 20, color, 2)
                                        
                                        # Add text with the call
                                        cv2.putText(
                                            annotated, 
                                            f"{call['call']['call']} ({call['call']['confidence']:.1f}%)", 
                                            (int(x) + 25, int(y)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                                        )
                                    
                                    st.image(annotated, caption=f"Frame {frame_idx}: {call['call']['call']} Call", 
                                            use_column_width=True)
                else:
                    st.info("No line calls available. Run advanced analysis to detect line calls.")
            
            with tab4:
                # Player Statistics
                st.subheader("Player Match Statistics")
                
                # Show shot statistics if we have data
                if st.session_state.shot_data:
                    shot_types = [shot['type'] for shot in st.session_state.shot_data if 'type' in shot]
                    
                    if shot_types:
                        # Count shot types
                        shot_counts = pd.Series(shot_types).value_counts()
                        
                        # Display key metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            forehand_count = shot_counts.get('forehand', 0)
                            st.metric("Forehand Shots", forehand_count)
                        
                        with col2:
                            backhand_count = shot_counts.get('backhand', 0)
                            st.metric("Backhand Shots", backhand_count)
                        
                        with col3:
                            total_shots = len(shot_types)
                            st.metric("Total Shots", total_shots)
                        
                        # More detailed statistics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Shot distribution visualization
                            fig, ax = plt.subplots()
                            shot_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                            ax.set_title('Shot Type Distribution')
                            ax.set_ylabel('')  # Hide the "None" ylabel
                            st.pyplot(fig)
                        
                        with col2:
                            # Shot speed comparison
                            shot_speeds = [(shot['type'], shot['speed']) for shot in st.session_state.shot_data 
                                         if 'type' in shot and 'speed' in shot]
                            
                            if shot_speeds:
                                df_speeds = pd.DataFrame(shot_speeds, columns=['Type', 'Speed'])
                                avg_speeds = df_speeds.groupby('Type')['Speed'].mean().reset_index()
                                
                                fig, ax = plt.subplots()
                                sns.barplot(x='Type', y='Speed', data=avg_speeds, ax=ax)
                                ax.set_title('Average Shot Speed by Type')
                                ax.set_ylabel('Speed (pixels/frame)')
                                st.pyplot(fig)
                        
                        # Court coverage and positioning
                        st.markdown("### Court Coverage")
                        
                        if st.session_state.player_positions:
                            # Create heatmap of player positions
                            all_positions = [player['position'] for player in st.session_state.player_positions 
                                           if 'position' in player]
                            
                            if all_positions:
                                # Create position arrays
                                x_positions = [pos[0] for pos in all_positions]
                                y_positions = [pos[1] for pos in all_positions]
                                
                                # Reference frame for court visualization
                                if st.session_state.frames:
                                    ref_frame = st.session_state.frames[len(st.session_state.frames) // 2]
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.imshow(ref_frame, alpha=0.6)
                                    
                                    # Create heatmap
                                    hb = ax.hexbin(x_positions, y_positions, gridsize=20, cmap='hot', alpha=0.7)
                                    ax.set_title('Player Court Coverage')
                                    fig.colorbar(hb, ax=ax, label='Position Density')
                                    
                                    st.pyplot(fig)
                        
                        # Serve statistics
                        st.markdown("### Serve Analysis")
                        
                        # Get serve data
                        serve_shots = [shot for shot in st.session_state.shot_data 
                                      if 'type' in shot and shot['type'] == 'serve']
                        
                        if serve_shots:
                            # Calculate some serve metrics (would be more comprehensive with real data)
                            serve_speeds = [shot.get('speed', 0) for shot in serve_shots]
                            avg_serve_speed = sum(serve_speeds) / len(serve_speeds) if serve_speeds else 0
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Serves", len(serve_shots))
                            
                            with col2:
                                st.metric("Avg. Serve Speed", f"{avg_serve_speed:.1f}")
                            
                            with col3:
                                # Simulating this metric
                                st.metric("First Serve %", "65%")
                else:
                    st.info("No player statistics available. Run advanced analysis to generate statistics.")

if __name__ == "__main__":
    main()
