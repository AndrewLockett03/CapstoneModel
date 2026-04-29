# Machine Learning Framework for Audio Declipping
### Andrew Lockett
### Capstone Project for TXST SRT 2026

This project outlines a multistage process of clipping detection, tagging, and machine learning to repair audio that has 
been damaged by digital or analog clipping.

### Setup

Download VCTK dataset from kagglehub:

```python dataset_download.py```

Run the training script:

```python train.py```


### Using your trained model

The script declip.py provides the framework for repairing a damaged audio file of your choosing

Run from command line using:
```python test.py --checkpoint ./checkpoints/best.pt --audio [PATH] --output_dir [OUT_DIR]```
