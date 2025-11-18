[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17577177.svg)](https://doi.org/10.5281/zenodo.17577177)

# AURA: Development and Validation of an Augmented Unplanned Removal Alert System using Synthetic ICU Videos

*AURA is an innovative vision-based system developed for the early detection of unplanned extubation (UE) risk in intensive care units (ICUs), featuring a new approach to synthesizing video datasets for model development and validation.*

# Abstract

Unplanned extubation (UE)—the unintended removal of an airway tube—remains a critical patient safety concern in intensive care units (ICUs), often leading to severe complications or death. 
Real-time UE detection has been limited, largely due to the ethical and privacy challenges of obtaining annotated ICU video data. 
We propose Augmented Unplanned Removal Alert (AURA), a vision-based risk detection system developed and validated entirely on a fully synthetic video dataset. 
By leveraging text-to-video diffusion, we generated diverse and clinically realistic ICU scenarios capturing a range of patient behaviors and care contexts. 
The system applies pose estimation to identify two high-risk movement patterns: collision, defined as hand entry into spatial zones near airway tubes, and agitation, quantified by the velocity of tracked anatomical keypoints. 
Expert assessments confirmed the realism of the synthetic data, and performance evaluations showed high accuracy for collision detection and moderate performance for agitation recognition.
This work demonstrates a novel pathway for developing privacy-preserving, reproducible patient safety monitoring systems with potential for deployment in intensive care settings.



# Setup


## Quick Start


```bash
conda env create -f environment.yml
conda activate AURA

# Process sample videos from assets directory
python main.py

# Process with video display
python main.py --show-video
```

## Dataset

The full dataset used for development and validation is available at [Zenodo](https://doi.org/10.5281/zenodo.17577177).

**Parameter Tuning**: The system parameters (collision detection thresholds, agitation detection thresholds, etc.) have been optimized by an experienced ICU nurse using the tuning set from AURA_dataset. The default parameters in the code reflect these expert-tuned values.

## File Structure

```
AURA/
├── main.py                
├── environment.yml        
├── README.md             
├── LICENSE               
├── .gitignore             
├── src/                   
│   ├── __init__.py
│   ├── collision.py       
│   ├── agitation.py       
│   ├── video_utils.py     
│   ├── alarm_logger.py   
│   └── evaluation.py      
├── assets/                # Input video files (sample videos)
│   ├── sample1.mp4        
│   └── sample2.mp4
├── AURA_dataset/          # AURA dataset (download from Zenodo)
│   ├── annotations/
│   │   └── expert_annotations.csv  # Expert annotations for evaluation
│   ├── metadata/
│   │   └── video_metadata.csv
│   ├── tuning/            # Tuning set videos (tuning_01.mp4 ~ tuning_12.mp4)
│   └── test/              # Test set videos (test_01.mp4 ~ test_63.mp4)
└── output/                # Processed video output
    ├── *_processed.mp4    # Processed videos
    ├── alarm_log.csv      # Alarm event logs
    └── performance_report.md  # Performance evaluation report (only for test set with --evaluate)
```

## Performance Evaluation

Performance evaluation is only available for the test set from AURA_dataset.

1. **Download AURA_dataset**: Download from [Zenodo](https://doi.org/10.5281/zenodo.17577177) and extract to project root

2. **Run evaluation on test set**:
   ```bash
   python main.py --dataset test --evaluate
   ```

3. **Results**: The evaluation report will be saved to `output/performance_report.md`

### Evaluation Metrics

The evaluation report includes:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **Error Analysis**: False positives and false negatives

### Command Line Options

- `--dataset {sample,test}`: Dataset to use (default: `sample`)
  - `sample`: Use sample videos from `assets/` directory (for demonstration only, no evaluation)
  - `test`: Use test set from `AURA_dataset/test/` (evaluation available)
  
- `--dataset-path PATH`: Path to AURA_dataset directory (default: `AURA_dataset`)
- `--evaluate`: Enable performance evaluation (only available for `test` dataset).
- `--show-video`: Display video window during processing

# License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

# Questions
If you have any questions, contact us at [junhyuk.sam@gmail.com](mailto:junhyuk.sam@gmail.com)


# Citation

If you use AURA in your research, please cite:
```bibtex
@article{seo2025aura,
      title={AURA: Development and Validation of an Augmented Unplanned Removal Alert System using Synthetic ICU Videos},
      author={Seo, Junhyuk and Moon, Hyeyoon and Jung, Kyu-Hwan and Oh, Namkee and Kim, Taerim},
      journal={arXiv preprint arXiv:2511.12241},
      year={2025}
    }
```