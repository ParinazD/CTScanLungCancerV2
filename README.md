# CTScanLungCancerV2
lung cancer detection from CT scans using a CNN model. Trained on the LIDC-IDRI dataset to classify pulmonary nodules as benign or malignant.

# **Feeding raw HU values directly into a Neural Network creates two significant technical hurdles:**
# 1. *Exploding Gradients:* 'Neural networks are mathematically optimized to process data
#  in a small, constrained range (typically $[0.0, 1.0]$ or $[-1.0, 1.0]$). Using large, 
# high-variance integers like $+400$ or $-1000$ can cause "exploding gradients" during 
# backpropagation, making the model's training unstable or preventing it from converging at all.
# 2. *Filtering Irrelevant Data:* A CT scan captures the entire density spectrum, from empty air to dense cortical bone. When screening for lung cancer, features like rib bones ($+700$ HU) are irrelevant noise that can distract the model.
