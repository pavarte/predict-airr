# predict-airr baseline (template-compatible)

Implements `ImmuneStatePredictor` with a simple, reliable baseline:
- Features: 3-mer frequencies on `junction_aa`, V/J usage, CDR3 length stats, Shannon diversity.
- Model: Elastic-net Logistic Regression with isotonic calibration.
- Ranking: Fisher's exact test over sequence presence per repertoire.

## How to use
Place `submission/predictor.py` into the official template repo (replacing the placeholder).
Make sure `requirements.txt` reflects below pins.

Then run:
```
python3 -m submission.main --train_dir /path/to/train_datasets --test_dir /path/to/test_datasets --out_dir /path/to/output --n_jobs 4 --device cpu
```

Outputs:
- predictions per test dataset: `predictions.csv` files
- ranked sequences per training dataset: `ranked_sequences.csv` files
- final `submissions.csv` is built by the template (if enabled).
