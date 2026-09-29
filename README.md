# Multimodal Sentiment Analysis

A lightweight multimodal sentiment analysis project that combines text and image information from social media-style posts. The project uses a simple feature-fusion approach that can run on a CPU-only machine.

## Project Overview

This project demonstrates how text and visual features can be combined to predict sentiment. Instead of using a large vision-language model, it uses compact pretrained/lightweight components so the complete workflow is practical on a normal laptop.

**Dataset:** [MVSA-Single](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-media/)  
**Approach:** TF-IDF text features + lightweight image color/edge features + Logistic Regression fusion.

The notebook downloads only the required public files when possible and stores them under `data/`.

## Repository Structure

```text
Multimodal-Sentiment-Analysis/
├── data/
│   └── README.md
├── models/
│   └── README.md
├── Multimodal_Sentiment_Analysis.ipynb
├── app.py
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
├── CHANGELOG.md
└── README.md
```

No `src/`, preprocessing module, or extra training script is used. The notebook contains the complete experiment and training workflow.

## How It Works

1. Load social media posts containing text and associated images.
2. Convert text into TF-IDF features.
3. Extract compact image statistics such as color histograms and edge density.
4. Normalize both modalities.
5. Concatenate the text and image representations.
6. Train a lightweight Logistic Regression classifier.
7. Compare text-only, image-only, and fused predictions.
8. Save the trained artifacts for the Flask demo.

## Run the Notebook

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `Multimodal_Sentiment_Analysis.ipynb` and run the cells from top to bottom.

## Run the Web App

After running the notebook and creating the model artifacts:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The web app accepts text and an optional image and displays the predicted sentiment.

## Interview Talking Points

- Why multimodal learning can capture information missed by text alone.
- Early feature fusion versus independent modality models.
- Why TF-IDF is useful as a CPU-friendly baseline.
- Why image statistics are used instead of a large vision transformer on a CPU-only laptop.
- How the same architecture could later be upgraded with CLIP or other vision-language encoders.
- How to compare text-only, image-only, and fused models fairly.

## Profiles

**GitHub:** https://github.com/InfinitePraveen  
**LinkedIn:** https://www.linkedin.com/in/infinitepraveen/

## License

This project is intended for educational and portfolio use.
