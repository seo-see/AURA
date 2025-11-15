import pandas as pd
from typing import Dict, List
import os
from datetime import datetime

class PerformanceEvaluator:
    """Evaluates performance of alarm detection system"""
    
    def __init__(self, ground_truth_path: str = "ground_truth.csv"):
        self.ground_truth_path = ground_truth_path
        self.gt_data = None
        self.is_expert_annotations = False
        self.load_ground_truth()
    
    def load_ground_truth(self):
        """Load ground truth data"""
        if self.ground_truth_path is None:
            print("Ground Truth path not specified.")
            self.gt_data = None
            return
            
        try:
            self.gt_data = pd.read_csv(self.ground_truth_path)
            
            if 'consensus_collision' in self.gt_data.columns and 'consensus_agitation' in self.gt_data.columns:
                self.is_expert_annotations = True
                print(f"Expert annotations loaded: {len(self.gt_data)} videos")
            elif 'GT_collision' in self.gt_data.columns and 'GT_agitation' in self.gt_data.columns:
                self.is_expert_annotations = False
                print(f"Ground Truth loaded: {len(self.gt_data)} videos")
            else:
                print(f"Warning: Unknown ground truth format in {self.ground_truth_path}")
                print(f"Expected columns: consensus_collision/consensus_agitation or GT_collision/GT_agitation")
                self.gt_data = None
        except FileNotFoundError:
            print(f"Ground Truth file not found: {self.ground_truth_path}")
            self.gt_data = None
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            self.gt_data = None
    
    def load_alarm_log(self, log_path: str) -> pd.DataFrame:
        """Load alarm log data"""
        try:
            log_data = pd.read_csv(log_path)
            print(f"Alarm log loaded: {len(log_data)} frames")
            return log_data
        except FileNotFoundError:
            print(f"Alarm log file not found: {log_path}")
            return pd.DataFrame()
    
    def extract_video_alarms(self, log_data: pd.DataFrame) -> Dict[str, Dict]:
        """Extract alarm status per video from summary CSV"""
        video_alarms = {}
        
        for _, row in log_data.iterrows():
            video_id = str(row['video_id']).strip()
            
            # Handle boolean values (may be string "True"/"False" or boolean)
            collision_detected = row.get('collision_detected', False)
            if isinstance(collision_detected, str):
                collision_detected = collision_detected.upper() == 'TRUE'
            else:
                collision_detected = bool(collision_detected)
            
            agitation_detected = row.get('agitation_detected', False)
            if isinstance(agitation_detected, str):
                agitation_detected = agitation_detected.upper() == 'TRUE'
            else:
                agitation_detected = bool(agitation_detected)
            
            video_alarms[video_id] = {
                'collision_detected': collision_detected,
                'agitation_detected': agitation_detected,
                'total_frames': int(row.get('total_frames', 0)),
                'collision_frames': int(row.get('collision_frames', 0)),
                'agitation_frames': int(row.get('agitation_frames', 0))
            }
        
        return video_alarms
    
    def calculate_metrics(self, predicted: List[bool], actual: List[bool]) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(predicted) != len(actual):
            raise ValueError("Predicted and actual lists have different lengths")
        
        tp = sum(1 for p, a in zip(predicted, actual) if p and a)
        tn = sum(1 for p, a in zip(predicted, actual) if not p and not a)
        fp = sum(1 for p, a in zip(predicted, actual) if p and not a)
        fn = sum(1 for p, a in zip(predicted, actual) if not p and a)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }
    
    def evaluate_performance(self, log_path: str) -> Dict:
        """Evaluate overall performance"""
        if self.gt_data is None:
            print("Ground Truth data not available.")
            return {}
        
        log_data = self.load_alarm_log(log_path)
        if log_data.empty:
            return {}
        
        video_alarms = self.extract_video_alarms(log_data)
        results = []
        collision_predicted = []
        collision_actual = []
        agitation_predicted = []
        agitation_actual = []
        
        for _, row in self.gt_data.iterrows():
            # Get video_id from ground truth
            gt_video_id = str(row['video_id']).strip()
            
            # Skip tuning set videos - they are not for evaluation
            if self.is_expert_annotations and gt_video_id.startswith('tuning_'):
                continue
            
            # Match video_id format: 
            # - For expert_annotations: test_01, tuning_01 etc. (already in correct format)
            # - For old format: aura_01 (needs conversion)
            # - From alarm log: video_id might be test_01, tuning_01, sample1, etc.
            if self.is_expert_annotations:
                # expert_annotations.csv uses test_01, tuning_01 format
                video_id_key = gt_video_id
            else:
                # Old format: convert to aura_XX format
                try:
                    video_num = int(gt_video_id)
                    video_id_key = f"aura_{video_num:02d}"
                except (ValueError, TypeError):
                    video_id_key = gt_video_id
            
            collision_pred = False
            agitation_pred = False
            
            if video_id_key in video_alarms:
                collision_pred = video_alarms[video_id_key].get('collision_detected', False)
                agitation_pred = video_alarms[video_id_key].get('agitation_detected', False)
            else:
                for log_video_id in video_alarms.keys():
                    if log_video_id.startswith(gt_video_id) or gt_video_id.startswith(log_video_id):
                        collision_pred = video_alarms[log_video_id].get('collision_detected', False)
                        agitation_pred = video_alarms[log_video_id].get('agitation_detected', False)
                        break
            
            if self.is_expert_annotations:
                collision_actual_val = bool(row.get('consensus_collision', False))
                agitation_actual_val = bool(row.get('consensus_agitation', False))
                if isinstance(collision_actual_val, str):
                    collision_actual_val = collision_actual_val.upper() == 'TRUE'
                if isinstance(agitation_actual_val, str):
                    agitation_actual_val = agitation_actual_val.upper() == 'TRUE'
            else:
                collision_actual_val = bool(row.get('GT_collision', False))
                agitation_actual_val = bool(row.get('GT_agitation', False))
            
            result = {
                'video_id': gt_video_id,
                'collision_predicted': collision_pred,
                'collision_actual': collision_actual_val,
                'agitation_predicted': agitation_pred,
                'agitation_actual': agitation_actual_val,
                'collision_correct': collision_pred == collision_actual_val,
                'agitation_correct': agitation_pred == agitation_actual_val
            }
            results.append(result)
            
            collision_predicted.append(collision_pred)
            collision_actual.append(collision_actual_val)
            agitation_predicted.append(agitation_pred)
            agitation_actual.append(agitation_actual_val)
        
        collision_metrics = self.calculate_metrics(collision_predicted, collision_actual)
        agitation_metrics = self.calculate_metrics(agitation_predicted, agitation_actual)
        
        return {
            'results': results,
            'collision_metrics': collision_metrics,
            'agitation_metrics': agitation_metrics,
            'total_videos': len(results)
        }
    
    def generate_report(self, log_path: str, output_path: str = None) -> str:
        """Generate performance evaluation report"""
        evaluation = self.evaluate_performance(log_path)
        
        if not evaluation:
            return "Evaluation cannot be performed."
        
        report = []
        report.append("# AURA Alarm Detection System Performance Evaluation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## 1. Overall Performance Summary")
        report.append("")
        
        collision_metrics = evaluation['collision_metrics']
        report.append("### Collision Detection Performance")
        report.append(f"- **Accuracy**: {collision_metrics['accuracy']:.3f}")
        report.append(f"- **Precision**: {collision_metrics['precision']:.3f}")
        report.append(f"- **Recall**: {collision_metrics['recall']:.3f}")
        report.append(f"- **F1-Score**: {collision_metrics['f1_score']:.3f}")
        report.append("")
        
        agitation_metrics = evaluation['agitation_metrics']
        report.append("### Agitation Detection Performance")
        report.append(f"- **Accuracy**: {agitation_metrics['accuracy']:.3f}")
        report.append(f"- **Precision**: {agitation_metrics['precision']:.3f}")
        report.append(f"- **Recall**: {agitation_metrics['recall']:.3f}")
        report.append(f"- **F1-Score**: {agitation_metrics['f1_score']:.3f}")
        report.append("")
        
        report.append("## 2. Confusion Matrix")
        report.append("")
        
        report.append("### Collision Detection Confusion Matrix")
        report.append("| | Predicted: No | Predicted: Yes |")
        report.append("|---|---|---|")
        report.append(f"| **Actual: No** | {collision_metrics['tn']} (TN) | {collision_metrics['fp']} (FP) |")
        report.append(f"| **Actual: Yes** | {collision_metrics['fn']} (FN) | {collision_metrics['tp']} (TP) |")
        report.append("")
        
        report.append("### Agitation Detection Confusion Matrix")
        report.append("| | Predicted: No | Predicted: Yes |")
        report.append("|---|---|---|")
        report.append(f"| **Actual: No** | {agitation_metrics['tn']} (TN) | {agitation_metrics['fp']} (FP) |")
        report.append(f"| **Actual: Yes** | {agitation_metrics['fn']} (FN) | {agitation_metrics['tp']} (TP) |")
        report.append("")
        
        report.append("## 3. Per-Video Detailed Results")
        report.append("")
        
        report.append("### Collision Detection Results")
        report.append("| Video ID | Predicted | Actual | Result |")
        report.append("|---|---|---|---|")
        for result in evaluation['results']:
            status = "✓" if result['collision_correct'] else "✗"
            video_id_str = str(result['video_id'])
            report.append(f"| {video_id_str} | {'Yes' if result['collision_predicted'] else 'No'} | {'Yes' if result['collision_actual'] else 'No'} | {status} |")
        report.append("")
        
        report.append("### Agitation Detection Results")
        report.append("| Video ID | Predicted | Actual | Result |")
        report.append("|---|---|---|---|")
        for result in evaluation['results']:
            status = "✓" if result['agitation_correct'] else "✗"
            video_id_str = str(result['video_id'])
            report.append(f"| {video_id_str} | {'Yes' if result['agitation_predicted'] else 'No'} | {'Yes' if result['agitation_actual'] else 'No'} | {status} |")
        report.append("")
        
        report.append("## 4. Error Analysis")
        report.append("")
        
        collision_errors = [r for r in evaluation['results'] if not r['collision_correct']]
        if collision_errors:
            report.append("### Collision Detection Errors")
            report.append("| Video ID | Predicted | Actual | Error Type |")
            report.append("|---|---|---|---|")
            for error in collision_errors:
                error_type = "False Positive" if error['collision_predicted'] and not error['collision_actual'] else "False Negative"
                video_id_str = str(error['video_id'])
                report.append(f"| {video_id_str} | {'Yes' if error['collision_predicted'] else 'No'} | {'Yes' if error['collision_actual'] else 'No'} | {error_type} |")
        else:
            report.append("### Collision Detection Errors: None")
        report.append("")
        
        agitation_errors = [r for r in evaluation['results'] if not r['agitation_correct']]
        if agitation_errors:
            report.append("### Agitation Detection Errors")
            report.append("| Video ID | Predicted | Actual | Error Type |")
            report.append("|---|---|---|---|")
            for error in agitation_errors:
                error_type = "False Positive" if error['agitation_predicted'] and not error['agitation_actual'] else "False Negative"
                video_id_str = str(error['video_id'])
                report.append(f"| {video_id_str} | {'Yes' if error['agitation_predicted'] else 'No'} | {'Yes' if error['agitation_actual'] else 'No'} | {error_type} |")
        else:
            report.append("### Agitation Detection Errors: None")
        report.append("")
        
        report.append("---")
        
        report_text = "\n".join(report)
        
        if output_path is None:
            output_path = os.path.join("output", "performance_report.md")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Performance evaluation report generated: {output_path}")
        return report_text

def main():
    """Run performance evaluation"""
    evaluator = PerformanceEvaluator()
    
    log_path = "output/alarm_log.csv"
    
    if not os.path.exists(log_path):
        print(f"Alarm log file not found: {log_path}")
        print("Please run main.py first to process videos and generate alarm log.")
        return
    
    report = evaluator.generate_report(log_path)
    print("\n" + "="*50)
    print("Performance evaluation completed!")
    print("="*50)

if __name__ == "__main__":
    main()
