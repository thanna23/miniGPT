# miniGPT

MiniGPT is a simplified GPT-like model for generating text. 
The objective of this project was to understand better how such models work, how to improve their training process.

## Overview :
### Hyperparameters :
    - Define training data 
    - Set model's parameters

### Tokenization : 
    - Character-Level encoder/decoder

### Data Preparation :
    - Encode dataset into integer sequence
    - Generate training batches
   
### Model Component : 
    - Positional & Token embedding Layer
    - Self-Attention mechanism
    - Tranformer Blocks
    - miniGPT model architecture

### Training Setup :
    - Initiate the model
    - Run the training Loop
    - Load and save model weights

### Text Generation : 
    - Generate text starting from a prompt

## Requirements :

- Python 3.8+
- TensorFlow 2.9+ 
- NumP
  
## Intallation & Virtual environment :

1. Create a virtual environment:
    python -m venv tf-env

2. Activate it :
    .\tf-env\Scripts\Activate.ps1

3. If permission error : 
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

4. Install TensirFlow :
    pip install tensorflow numpy

  