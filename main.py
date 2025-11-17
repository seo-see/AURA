import cv2
import mediapipe as mp
import os
import time
from src.collision import draw_landmarks_on_image, CollisionManager, CollisionConfig
from src.video_utils import get_video_info
from src.agitation import MovementAnalyzer
from src.alarm_logger import AlarmLogger

def process_video(input_path, show_video=False, enable_evaluation=False, ground_truth_path=None):
    video_start_time = time.time()
    
    # Get video information
    width, height, fps, total_frames = get_video_info(input_path)
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    
    # Set up output video file
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract base name from input filename (without extension)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_processed.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    )
    
    # Collision detection configuration
    config = CollisionConfig()
    
    # Initialize collision manager
    collision_manager = CollisionManager(config)
    
    # === Agitation (Movement) Analysis Parameters ===
    agitation_keypoints = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Head, face parts
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,  # Shoulders, elbows, hands, fingers
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32  # Hips, knees, ankles, feet
    ]
    agitation_window_size = 5                      # Number of frames for cumulative movement calculation
    agitation_threshold = 0.18                     # Speed threshold
    agitation_on = True                            # Analysis on/off
    analyzer = MovementAnalyzer(
        keypoint_indices=agitation_keypoints,
        window_size=agitation_window_size,
        speed_threshold=agitation_threshold,
        analyze_movement=agitation_on
    )
    
    # Initialize alarm logger
    alarm_logger = AlarmLogger()
    alarm_logger.start_video(input_path)
    
    # Initialize performance evaluator (only if evaluation is enabled)
    evaluator = None
    if enable_evaluation:
        from src.evaluation import PerformanceEvaluator
        evaluator = PerformanceEvaluator(ground_truth_path=ground_truth_path)
    
    # Variables to store previous frame values (for text persistence)
    prev_lh_text = None
    prev_rh_text = None
    prev_mean_v = 0.0
    prev_peak_v = 0.0
    prev_cum_v = 0.0
    prev_agitation_alarm = False
    
    # Variables for alarm message persistence (2 seconds)
    alarm_hold_duration = 2.0  # seconds
    last_collision_alarm_time = None  # None until first alarm is detected
    last_agitation_alarm_time = None  # None until first alarm is detected
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = pose.process(rgb_frame)
        
        # Draw landmarks and collision layers
        annotated_frame, lh_text, rh_text = draw_landmarks_on_image(
            rgb_frame, results, collision_manager, config
        )
        
        # === Movement (agitation) analysis ===
        mean_v = prev_mean_v
        peak_v = prev_peak_v
        cum_v = prev_cum_v
        current_agitation_alarm = prev_agitation_alarm
        
        if results.pose_landmarks:
            mean_v, peak_v, cum_v, current_agitation_alarm = analyzer.update(results.pose_landmarks.landmark)
            # Update previous values when keypoints are detected
            prev_mean_v = mean_v
            prev_peak_v = peak_v
            prev_cum_v = cum_v
            prev_agitation_alarm = current_agitation_alarm
            # Update text values when available
            if lh_text:
                prev_lh_text = lh_text
            if rh_text:
                prev_rh_text = rh_text
        
        # Display values (scores, velocity) - always display, using previous values if keypoints not detected
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 255)
        w = annotated_frame.shape[1]
        
        # Display LH, RH scores (fixed position) - use previous values if current is None
        display_lh_text = lh_text if lh_text is not None else prev_lh_text
        display_rh_text = rh_text if rh_text is not None else prev_rh_text
        if display_lh_text:
            cv2.putText(annotated_frame, display_lh_text, (w-400, 50), font, font_scale, color, thickness)
        if display_rh_text:
            cv2.putText(annotated_frame, display_rh_text, (w-400, 90), font, font_scale, color, thickness)
        
        # Display velocity information (fixed position) - always display
        velocity_text = analyzer.get_velocity_text(mean_v, peak_v, cum_v)
        cv2.putText(annotated_frame, velocity_text, (w-400, 130), font, font_scale, color, thickness)
        
        current_time = time.time() - video_start_time
        
        # Update alarm trigger times when alarms are detected
        if collision_manager.is_colliding:
            last_collision_alarm_time = current_time
        if current_agitation_alarm:
            last_agitation_alarm_time = current_time
        
        # Collision detection message (fixed position) - display for 2 seconds after last detection
        if last_collision_alarm_time is not None and current_time - last_collision_alarm_time < alarm_hold_duration:
            collision_color = (255, 0, 0)
            cv2.putText(annotated_frame, "COLLISION DETECTED!", (w-400, 170), font, 1.0, collision_color, 3)
        
        # Excessive behavior detection message (fixed position) - display for 2 seconds after last detection
        if last_agitation_alarm_time is not None and current_time - last_agitation_alarm_time < alarm_hold_duration:
            agitation_color = (255, 255, 0)
            cv2.putText(annotated_frame, "AGITATION DETECTED!", (w-400, 210), font, 1.0, agitation_color, 3)
        alarm_logger.log_frame(
            frame_number=collision_manager.frame_count,
            time_seconds=current_time,
            collision_alarm=collision_manager.is_colliding,
            agitation_alarm=current_agitation_alarm,
            collision_score=0.0,
            agitation_velocity=0.0
        )
        
        # Convert RGB back to BGR
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Save frame
        out.write(annotated_frame)
        
        if show_video:
            cv2.imshow('Pose Detection', annotated_frame)
        
        collision_manager.frame_count += 1
        if collision_manager.frame_count % 10 == 0:
            print(f"Processing frame: {collision_manager.frame_count}/{total_frames}")
        
        if show_video and cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    pose.close()
    
    if show_video:
        cv2.destroyAllWindows()
    
    # End video processing
    alarm_logger.end_video()
    
    # Evaluate performance (only if evaluation is enabled)
    if enable_evaluation and evaluator is not None:
        log_path = os.path.join("output", "alarm_log.csv")
        if os.path.exists(log_path):
            try:
                evaluator.evaluate_performance(log_path)
                print(f"Performance evaluation completed.")
            except Exception as e:
                print(f"Error during performance evaluation: {e}")
        else:
            print(f"Alarm log file not found: {log_path}")
    
    print(f"Video processing complete. Processed video saved to {output_path}")

if __name__ == "__main__":
    import argparse
    import glob
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='AURA: Augmented Unplanned Removal Alert')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Enable performance evaluation (requires expert_annotations.csv for AURA_dataset)')
    parser.add_argument('--show-video', action='store_true',
                       help='Display video during processing')
    parser.add_argument('--dataset', choices=['sample', 'test'], default='sample',
                       help='Dataset to use: sample (assets) or test (from AURA_dataset)')
    parser.add_argument('--dataset-path', type=str, default='AURA_dataset',
                       help='Path to AURA_dataset directory (default: AURA_dataset)')
    args = parser.parse_args()
    
    # Get video files based on dataset selection
    video_files = []
    
    if args.dataset == 'sample':
        # Default: Use sample videos from assets directory
        assets_dir = "assets"
        sample_patterns = ["sample*.mp4", "*.mp4"]
        for pattern in sample_patterns:
            video_files.extend(glob.glob(os.path.join(assets_dir, pattern)))
        video_files = sorted(list(set(video_files)))  # Remove duplicates and sort
        
        if not video_files:
            print(f"No sample videos found in '{assets_dir}' directory.")
            print("Looking for sample*.mp4 or *.mp4 files")
            exit(1)
    
    elif args.dataset == 'test':
        # Use AURA_dataset test set
        dataset_dir = os.path.join(args.dataset_path, "test")
        
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory '{dataset_dir}' not found.")
            print(f"Please ensure AURA_dataset is downloaded and placed in the project root.")
            print(f"Or specify the correct path with --dataset-path option.")
            exit(1)
        
        # Get all video files in the dataset directory
        video_files = sorted(glob.glob(os.path.join(dataset_dir, "*.mp4")))
        
        if not video_files:
            print(f"No videos found in '{dataset_dir}' directory.")
            exit(1)
    
    print(f"Found {len(video_files)} test video(s) to process:")
    for i, video_path in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(video_path)}")
    print()
    
    # Check if evaluation is requested for sample videos (not supported)
    if args.evaluate and args.dataset == 'sample':
        print("Warning: Performance evaluation is not available for sample videos.")
        print("Sample videos are for demonstration purposes only.")
        print("Please use --dataset test --evaluate for performance evaluation.")
        print("Evaluation will be disabled for sample videos.")
        print()
        args.evaluate = False  # Disable evaluation for sample videos
    
    if args.evaluate:
        print("Performance evaluation is enabled.")
        annotations_path = os.path.join(args.dataset_path, "annotations", "expert_annotations.csv")
        print(f"Note: This requires expert_annotations.csv at: {annotations_path}")
        print()
    
    # Determine ground truth path for evaluation
    ground_truth_path = None
    if args.evaluate and args.dataset == 'test':
        ground_truth_path = os.path.join(args.dataset_path, "annotations", "expert_annotations.csv")
    
    # Process each video file
    for i, video_path in enumerate(video_files, 1):
        print(f"Processing video {i}/{len(video_files)}: {os.path.basename(video_path)}")
        print("-" * 60)
        try:
            process_video(video_path, show_video=args.show_video, 
                         enable_evaluation=args.evaluate, ground_truth_path=ground_truth_path)
            print(f"✓ Completed: {os.path.basename(video_path)}")
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(video_path)}: {e}")
        print()
    
    print("All test videos processed!")
    print(f"Check the 'output' directory for processed videos and alarm logs.")
    
    # Generate evaluation report if evaluation was enabled
    if args.evaluate:
        try:
            from src.evaluation import PerformanceEvaluator
            evaluator = PerformanceEvaluator(ground_truth_path=ground_truth_path)
            log_path = os.path.join("output", "alarm_log.csv")
            if os.path.exists(log_path):
                print("\n" + "="*60)
                print("Generating performance evaluation report...")
                print("="*60)
                evaluator.generate_report(log_path)
            else:
                print(f"\nAlarm log file not found: {log_path}")
        except Exception as e:
            print(f"\nError generating performance evaluation report: {e}")
