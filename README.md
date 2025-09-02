## MedReShaper Introductions
MedReShaper: An AI-driven Workflow for 3D Anatomical Shape Modeling with Multi-Ethnic Clinical Validations in Orbital Surgery

MedReShaper is an AI-based system for automatically completing 3D anatomic shapes in clinical settings. It introduces a novel fishbone-structured network architecture, ASM-Net, capable of reconstructing anatomically plausible shapes from defective or incomplete medical shapes. The system is validated on clinical and multi-center datasets, demonstrating sub-millimeter accuracy and strong generalizability.

<img width="3482" height="2011" alt="image" src="https://github.com/user-attachments/assets/05009b81-fdf2-4782-ba55-25fd6024391b" />

## Repository Structure
├── BaseLineModels/ # Baseline models (U-Net, V-Net, nnU-Net for P-AM only)  
├── DataPreparation/ # Scripts for normalization, P-AM (1) and sub-AMs (7) generation  
├── SurfaceRec/ # Reconstruction codes: AM → Point Cloud → Surface mesh  
├── ASM-Net.py # Main ASM-Net architecture  
├── loss.py # Loss functions  
├── requirements.txt # Python dependencies  
├── train.py # Training entry point  
└── README.md # Documentation

## ASM-Net Framework

<img width="1440" height="473" alt="image" src="https://github.com/user-attachments/assets/8a069587-a5b2-4cf0-9e1a-fd3c67b8c3b1" />

## Key Features:

🧩 End-to-end pipeline from PCA-based perception Attention Maps (AMs)

🧠 ASM-Net architecture with multi-angle fusion and its latent reasoning

📦 Input & Output: 3D meshes or point clouds

🔬 Clinically validated on orbital reconstruction cases

🌍 Evaluated by Multi-ethnic, multi-source datasets

## Reference Models:
U-Net:   https://github.com/milesial/Pytorch-UNet

V-Net:   https://github.com/yingkaisha/keras-unet-collection

nnUnet:  https://github.com/MIC-DKFZ/nnUNet

## Training Example & Quick Start
Step 1: pip install -r requirements.txt

Step 2: Run codes in "DataPreparation" to normalize, prepare P-AM (1), and sub-AMs (7)

Step 3: Run train.py 

Step 4: Run codes in "SurfaceRec" and convert the model output from AMs → Point Clouds → Surface Meshes, and automatically get the best surface among multiple candidates generated from a single P-AM.

## Citation
If you use this code, please cite:  
@article{your2025medreshaper,
  title={MedReShaper: Multi-View Anatomical Shape Reconstruction from Incomplete Meshes},
  author={Your Name and Others},
  journal={Medical Image Analysis},
  year={2025}}
