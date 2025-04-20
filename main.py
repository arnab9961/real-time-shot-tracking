import os
import tempfile
import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import google.generativeai as genai
from dotenv import load_dotenv

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
    
    st.markdown("""
    ## Upload a tennis video for AI-powered analysis
    This application uses Gemini 2.5 to analyze tennis videos, identify player movements,
    track the ball, classify shots, and detect key events.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a tennis video", type=['mp4', 'mov', 'avi'])
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # Display the original video
            st.subheader("Original Video")
            st.video(video_path)
            
            # Process button
            if st.button("Analyze Video"):
                results = analyze_video(video_path)
                st.success("Analysis complete! Use the frame navigator to view results.")
    
    with col2:
        # Display analysis results if available
        if st.session_state.frames and st.session_state.analysis_results:
            st.subheader("Analysis Results")
            
            # Frame navigation
            st.subheader("Frame Navigation")
            selected_frame = st.slider("Select frame", 0, 
                                      st.session_state.total_frames - 1, 
                                      st.session_state.current_frame)
            st.session_state.current_frame = selected_frame
            
            # Display the selected frame
            frame = st.session_state.frames[selected_frame]
            
            # Check if this frame has analysis
            frame_analysis = ""
            if selected_frame in st.session_state.analysis_results:
                frame_analysis = st.session_state.analysis_results[selected_frame]['analysis']
                
                # Create annotated frame
                annotated_frame = create_annotated_frame(frame, frame_analysis)
                st.image(annotated_frame, caption=f"Frame {selected_frame}", use_column_width=True)
                
                # Show detailed analysis
                with st.expander("Detailed Analysis"):
                    st.markdown(frame_analysis)
            else:
                st.image(frame, caption=f"Frame {selected_frame} (No analysis available)", use_column_width=True)
                
            # Shot metrics (placeholder - would be derived from actual analysis)
            with st.expander("Shot Metrics"):
                st.markdown("### Shot Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Forehand Count", "12")
                with col2:
                    st.metric("Backhand Count", "8")
                with col3:
                    st.metric("Serve Speed Avg", "110 mph")
                
                # Sample chart for shot distribution
                fig, ax = plt.subplots()
                shots = ['Forehand', 'Backhand', 'Serve', 'Volley', 'Overhead']
                counts = [12, 8, 5, 3, 1]
                ax.bar(shots, counts)
                ax.set_ylabel('Count')
                ax.set_title('Shot Distribution')
                st.pyplot(fig)

if __name__ == "__main__":
    main()