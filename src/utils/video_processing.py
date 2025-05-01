import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import logging

def read_video(video_path):
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    return video_capture

def write_video(output_path, frame_width, frame_height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return out

def display_frame(frame, window_name='Video'):
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

def release_resources(cap, out):
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def display_video_with_ui(video_capture, process_frame):
    # Create a Tkinter window
    window = tk.Tk()
    window.title("Real-Time Video")

    # Create a label to display the video
    video_label = tk.Label(window)
    video_label.pack()

    def update_frame():
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Failed to read frame from video capture. Exiting...")
            video_capture.release()
            window.destroy()
            return

        # Debugging: Log that a frame is being processed
        logging.info("Processing a new frame...")

        # Process the frame
        try:
            frame = process_frame(frame)
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return

        # Convert the frame to an image format compatible with Tkinter
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the label with the new frame
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        except Exception as e:
            logging.error(f"Error converting frame for Tkinter: {e}")
            return

        # Schedule the next frame update
        window.after(10, update_frame)

    # Start the video loop
    update_frame()

    # Run the Tkinter main loop
    window.mainloop()

def display_video_with_opencv(video_capture, process_frame):
    # Get original video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame delay to match original video's FPS
    frame_delay = int(1000 / original_fps)
    
    # Create a window with original video dimensions
    cv2.namedWindow('Real-Time Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Real-Time Video', frame_width, frame_height)

    logging.info(f"Original video dimensions: {frame_width}x{frame_height}, FPS: {original_fps}")

    while True:
        start_time = cv2.getTickCount()  # Start timing for FPS control
        
        ret, frame = video_capture.read()
        if not ret:
            logging.info("End of video reached.")
            break

        # Process the frame while preserving original dimensions
        try:
            processed_frame = process_frame(frame.copy())
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            processed_frame = frame  # Show original frame if processing fails
        
        # Display the frame at its original size
        cv2.imshow('Real-Time Video', processed_frame)

        # Calculate elapsed time to maintain correct FPS
        elapsed_ms = (cv2.getTickCount() - start_time) * 1000 / cv2.getTickFrequency()
        wait_time = max(1, int(frame_delay - elapsed_ms))
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            logging.info("Playback stopped by user.")
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()