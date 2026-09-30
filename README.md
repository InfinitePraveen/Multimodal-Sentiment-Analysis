# Multimodal Sentiment Analysis

Predict the sentiment (positive / negative / neutral) of a social media post from **both** its text and its image.
The project compares single-modality models with early and late fusion, and comes with a small Flask app so the
model can be tried out live.

Skills touched: multimodal learning, vision-language features, fusion models, end-to-end ML pipeline, Flask.

I built this to run on an ordinary laptop: no GPU, and not much disk space.

## How it works

```
tweet text  -> clean -> TF-IDF ------------------------+
                                                       +--> fusion --> positive / negative / neutral
tweet image -> MobileNetV3-small (frozen) -> 576-d ----+
```

* Text features: TF-IDF with 1-2 word n-grams.
* Image features: a frozen, pretrained MobileNetV3-small (about 10 MB). Nothing is fine-tuned, it is one forward pass per image on CPU.
* Fusion strategies compared in notebook 03:
  1. text only
  2. image only
  3. early fusion: concatenate both vectors, then logistic regression or a small MLP
  4. late fusion: weighted average of the two per-modality probabilities
* The fusion model with the best validation macro-F1 is saved and used by the web app.
  If only text or only an image is given, the app falls back to the matching single model.

## Data

MVSA-Single (Niu et al., *Sentiment Analysis on Multi-View Social Data*, MMM 2016): about 5k tweets, each with one
image and a positive / negative / neutral label. The notebook downloads a community copy from the Hugging Face Hub
(`xwycyj/MVSA-Single`). The data is for research use and is not part of this repo.

The zip is ~211 MB. It is read directly from the zip, never extracted.

## Repository layout

```
notebooks/
  01_data_download_and_eda.ipynb      download, labels, cleaning, split, quick EDA
  02_image_features.ipynb             MobileNetV3-small embeddings for every image
  03_text_features_and_fusion.ipynb   TF-IDF, fusion models, comparison, save the best one
app.py                                Flask demo
templates/  static/                   page + css (plots and demo examples end up in static/ too)
data/  models/                        filled by the notebooks (git-ignored)
```

## Run it

Python 3.9 - 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

1. Run the notebooks in `notebooks/` in order: 01 -> 02 -> 03.
   Roughly: 01 is dominated by the 211 MB download, 02 takes a few minutes on a CPU, 03 takes a minute or two.
2. Start the app from the repo root:

```bash
python app.py
```

Open http://127.0.0.1:5000. The first start downloads the MobileNet weights (~10 MB) once.

### Keeping disk usage small

* `requirements.txt` points pip at the CPU-only PyTorch wheels. That avoids the multi-GB CUDA packages on Linux.
* Once notebook 02 has finished you can delete `data/data.zip` (211 MB). The app only needs `models/` and `static/`.
* Nothing else is large: features are a few MB, the saved model is a few MB.

## Results

Notebook 03 prints the comparison table and writes it to `models/metrics.json`; the app shows the same table,
a bar chart and the confusion matrix. Numbers depend on your split and library versions, so I do not
hard-code them here.

## Design choices and limitations

* **Why TF-IDF and not BERT / CLIP?** Both are much bigger downloads and slow on a CPU. TF-IDF + a frozen small CNN is
  a fair baseline that anyone can rerun. Swapping in sentence embeddings or CLIP features would only change the
  feature step; the fusion code stays the same.
* **Macro-F1 next to accuracy** because the neutral class is small.
* MVSA-Single has one annotator per tweet, so labels are noisy, and the dataset is small. Expect modest scores.
* An ImageNet CNN recognises objects, not moods; sarcasm and text embedded in images are not handled.
* The label mapping (0 = positive, 1 = negative, 2 = neutral) in notebook 01 was inferred from the data, see the note there.
* The model bundle is a joblib file, so use the same scikit-learn version for the notebooks and the app.

## Author

Praveen - [GitHub](https://github.com/InfinitePraveen) - [LinkedIn](https://www.linkedin.com/in/infinitepraveen)

Contributions are welcome, see [CONTRIBUTE.md](CONTRIBUTE.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).
Code is under the MIT licence; the dataset keeps its own terms.
