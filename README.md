# ZISVFM: Zero-Shot Object Instance Segmentation in Indoor Robotic Environments with Vision Foundation Models

## Abstract
Service robots operating in unstructured environments must effectively recognize and segment unknown objects to enhance their functionality. Traditional supervised learning-based segmentation techniques require extensive annotated datasets, impractical for the diversity of objects encountered in real-world scenarios. This paper introduces a novel approach, ZISVFM (Zero-Shot Instance Segmentation with Vision Foundation Models), which leverages the powerful zero-shot capabilities of the Segment Anything Model (SAM) integrated with explicit visual concepts from a self-supervised Vision Transformer (ViT). Our framework operates in three stages to generate precise object segmentation, validated on two benchmark datasets and in real-world environments, demonstrating its effectiveness and practical utility.

## Method Overview
The ZISVFM framework operates in three main stages:
1. **Initial Mask Proposal Generation**: Generate object-agnostic mask proposals using SAM from colorized depth images.
2. **Refinement of Proposals**: Refine these mask proposals by removing non-object masks based on size characteristics and integrating explicit visual concepts obtained from a self-supervised ViT.
3. **Final Object Segmentation**: Employ K-Medoids clustering to generate point prompts within the object proposal regions, guiding the SAM towards precise segmentation.

![Method Overview](path/to/method_overview_image.png)

## Requirements
To run the code, you will need the following libraries:
- PyTorch 1.x
- Transformers
- OpenCV

## Installation
Clone this repository and install the required packages:
```bash
git clone https://github.com/yourgithub/zisvfm.git
cd zisvfm
pip install -r requirements.txt

