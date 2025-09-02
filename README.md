# MedReShaper Introductions
MedReShaper: An AI-driven Workflow for 3D Anatomical Shape Modeling with Multi-Ethnic Clinical Validations in Orbital Surgery

MedReShaper is an AI-based system for automatically completing 3D anatomic shapes in clinical settings. It introduces a novel fishbone-structured network architecture, ASM-Net, capable of reconstructing anatomically plausible shapes from defective or incomplete medical shapes. The system is validated on clinical and multi-center datasets, demonstrating sub-millimeter accuracy and strong generalizability.

<img width="3482" height="2011" alt="image" src="https://github.com/user-attachments/assets/05009b81-fdf2-4782-ba55-25fd6024391b" />

# ASM-Net Framework

<img width="1440" height="473" alt="image" src="https://github.com/user-attachments/assets/8a069587-a5b2-4cf0-9e1a-fd3c67b8c3b1" />


# Key Features:

🧩 End-to-end pipeline from PCA-based perception to mesh reconstruction

🧠 ASM-Net architecture with attention-based latent reasoning

📦 Input & Output: 3D meshes or point clouds

🔬 Clinically validated on orbital reconstruction cases

🌍 Multi-ethnic, multi-source datasets supported

# Reference Models:
U-Net:   https://github.com/milesial/Pytorch-UNet

V-Net:   https://github.com/yingkaisha/keras-unet-collection

nnUnet:  https://github.com/MIC-DKFZ/nnUNet

# Training Example & Quick Start
Step 1: pip install -r requirements.txt

Step 2: Run codes in "DataPreparation" to normalize, prepare P-AM (1), and sub-AMs (7)

Step 3: python train.py 


