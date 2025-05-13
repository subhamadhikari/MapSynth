# Pix2Pix GAN for Satellite-to-Map Image Translation

This project implements a Pix2Pix GAN (Generative Adversarial Network) for image-to-image translation tasks, specifically converting satellite images into map-style images. The implementation is done using PyTorch, inspired by the original Pix2Pix paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004).

## Project Overview

- **Objective:** Convert input satellite images into corresponding map images using a conditional GAN framework.
- **Dataset:** Pix2Pix Maps.
![Sample Dataset](plots/sample.jpeg)

- **Model Architecture:**
  - **Generator:** U-Net-based encoder-decoder with skip connections.
  - **Discriminator:** PatchGAN that classifies whether local image patches are real or fake.

## Results
Training loss curves for both the Discriminator and Generator are plotted and saved inside the `plots/` folder.

### Loss Curves:
![Training Loss Curves](plots/loss.png)

