import csv
import os
from datetime import datetime
from typing import Dict

class AlarmLogger:
    """Logs alarm events to CSV file"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "alarm_log.csv")
        self.current_video = None
        self.alarm_events = []
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize log file with CSV header"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0:
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'video_id', 'total_frames', 'collision_detected', 'agitation_detected',
                    'collision_frames', 'agitation_frames', 
                    'collision_duration', 'agitation_duration', 'processing_date'
                ])
    
    def start_video(self, video_path: str):
        """Start processing a new video"""
        self.current_video = os.path.splitext(os.path.basename(video_path))[0]
        self.alarm_events = []
    
    def log_frame(self, frame_number: int, time_seconds: float, 
                  collision_alarm: bool, agitation_alarm: bool,
                  collision_score: float = 0.0, agitation_velocity: float = 0.0,
                  collision_duration: float = 0.0, agitation_duration: float = 0.0):
        """Track alarm status for each frame (stored in memory, written as summary at end)"""
        event = {
            'frame_number': frame_number,
            'time_seconds': round(time_seconds, 3),
            'collision_alarm': collision_alarm,
            'agitation_alarm': agitation_alarm,
            'collision_score': round(collision_score, 3),
            'agitation_velocity': round(agitation_velocity, 3),
            'collision_duration': round(collision_duration, 3),
            'agitation_duration': round(agitation_duration, 3)
        }
        
        self.alarm_events.append(event)
    
    def log_alarm_event(self, alarm_type: str, start_time: float, end_time: float, 
                       frame_start: int, frame_end: int, additional_info: Dict = None):
        """Log alarm event (start-end)"""
        event = {
            'type': alarm_type,
            'video_id': self.current_video,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'frame_start': frame_start,
            'frame_end': frame_end,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'additional_info': additional_info or {}
        }
        
        print(f"🚨 {alarm_type.upper()} Alarm: {start_time:.2f}s - {end_time:.2f}s "
              f"(Frames {frame_start}-{frame_end}, Duration: {event['duration']:.2f}s)")
    
    def get_video_summary(self) -> Dict:
        """Get alarm summary for current video"""
        if not self.alarm_events:
            return {
                'video_id': self.current_video,
                'total_frames': 0,
                'collision_alarms': 0,
                'agitation_alarms': 0,
                'collision_duration': 0.0,
                'agitation_duration': 0.0
            }
        
        collision_alarms = sum(1 for event in self.alarm_events if event['collision_alarm'])
        agitation_alarms = sum(1 for event in self.alarm_events if event['agitation_alarm'])
        
        collision_duration = self._calculate_alarm_duration('collision_alarm')
        agitation_duration = self._calculate_alarm_duration('agitation_alarm')
        
        return {
            'video_id': self.current_video,
            'total_frames': len(self.alarm_events),
            'collision_alarms': collision_alarms,
            'agitation_alarms': agitation_alarms,
            'collision_duration': collision_duration,
            'agitation_duration': agitation_duration
        }
    
    def _calculate_alarm_duration(self, alarm_type: str) -> float:
        """Calculate total duration of consecutive alarms"""
        if not self.alarm_events:
            return 0.0
        
        duration = 0.0
        in_alarm = False
        alarm_start = 0.0
        
        for event in self.alarm_events:
            if event[alarm_type]:
                if not in_alarm:
                    in_alarm = True
                    alarm_start = event['time_seconds']
            else:
                if in_alarm:
                    duration += event['time_seconds'] - alarm_start
                    in_alarm = False
        
        if in_alarm and self.alarm_events:
            duration += self.alarm_events[-1]['time_seconds'] - alarm_start
        
        return duration
    
    def print_summary(self):
        """Print alarm summary for current video"""
        summary = self.get_video_summary()
        print(f"\n=== {summary['video_id']} Alarm Summary ===")
        print(f"Total frames: {summary['total_frames']}")
        print(f"Collision alarms: {summary['collision_alarms']} frames ({summary['collision_duration']:.2f}s)")
        print(f"Agitation alarms: {summary['agitation_alarms']} frames ({summary['agitation_duration']:.2f}s)")
    
    def end_video(self):
        """End video processing and write summary to CSV"""
        if self.current_video:
            summary = self.get_video_summary()
            
            # Write video summary to CSV
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    summary['video_id'],
                    summary['total_frames'],
                    summary['collision_alarms'] > 0,  # collision_detected
                    summary['agitation_alarms'] > 0,  # agitation_detected
                    summary['collision_alarms'],  # collision_frames
                    summary['agitation_alarms'],  # agitation_frames
                    round(summary['collision_duration'], 3),
                    round(summary['agitation_duration'], 3),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
            
            self.print_summary()
            self.current_video = None
            self.alarm_events = []
