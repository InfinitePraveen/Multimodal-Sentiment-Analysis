# Contributing

Thanks for looking at this project. It is a small learning project, so the rules are light.

## Ways to help

* Report a bug (a notebook cell that fails, a wrong number, an app crash). Open an issue with your OS, Python version
  and the error message.
* Suggest or add an improvement, for example better text features, another fusion idea, or a cleaner web page.
* Fix typos or unclear explanations in the notebooks and README.

## Setting up

```bash
git clone https://github.com/InfinitePraveen/multimodal-sentiment-analysis.git
cd multimodal-sentiment-analysis
python -m venv .venv
pip install -r requirements.txt
```

Then run the notebooks in order (01, 02, 03) and start the app with `python app.py`.

## Ground rules

* Keep the repo simple. Work lives in the notebooks; `app.py` is the only Python file. Please do not add `src/` packages
  or helper scripts.
* Keep it light: the project should still run on a CPU-only laptop with little free disk space. No big model downloads
  by default; if you add one, make it optional and say how big it is.
* Clear notebook outputs before committing (Kernel -> Restart & Clear Output) so diffs stay readable.
* Do not commit data, the model bundle or other generated files (see `.gitignore`).
* If you change `clean_text` in notebook 01, change the copy in `app.py` too.

## Pull requests

1. Fork and create a branch, for example `fix-late-fusion-weight`.
2. Make one focused change and run the notebooks and the app once to be sure nothing broke.
3. Add a line under **Unreleased** in `CHANGELOG.md`.
4. Open the pull request and say what changed and why.

Be kind in issues and reviews.
