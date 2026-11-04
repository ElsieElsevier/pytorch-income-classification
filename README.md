# Income Classification with PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ElsieElsevier/pytorch-income-classification/blob/main/IncomeBracketStoppage.ipynb)

A PyTorch neural network that predicts whether a person earns more than $50K a year from census data. I built it as my first machine learning project to learn PyTorch, then came back to fix its bugs and test which improvements actually helped.

**[View the notebook](IncomeBracketStoppage.ipynb)** to see the full workflow with its saved results.

## Results

Scores on the 16,281-row test set, with every model using the same features and training data:

| Model | Accuracy | ROC-AUC | F1 (>50K) | Precision (>50K) | Recall (>50K) |
|---|---|---|---|---|---|
| Always predict `<=50K` | 76.38% | 0.500 | 0.000 | 0.000 | 0.000 |
| Logistic regression | 86.23% | 0.915 | 0.679 | 0.755 | 0.618 |
| **PyTorch network (mean of 5 seeds)** | **86.16%** | **0.916** | **0.678** | **0.752** | **0.618** |
| Gradient boosting (default settings) | 87.24% | 0.927 | 0.709 | 0.769 | 0.657 |

**What this shows:**

- The network clearly beats the majority-class baseline, but it finds only about 62% of people earning over $50K.
- It ties logistic regression, so with these features its extra complexity adds no measurable benefit.
- Gradient boosting scores highest, which is typical for tabular data like this. I kept PyTorch because learning it was the project's goal.

## What I learned

I tested each candidate improvement on its own, using a fixed validation split so that test data couldn't influence the choices. Scores are the mean of 3 seeds, and the baseline was 0.9170 ROC-AUC.

| Change | Validation ROC-AUC | Decision |
|---|---|---|
| Add quantile bins for numeric features | 0.9209 | Kept: biggest gain |
| Keep `?` as an `Unknown` category | 0.9180 | Kept: small gain |
| Lower dropout to 0.1 | 0.9169 | Kept: similar score, faster training |
| Use `BCEWithLogitsLoss` | 0.9171 | Kept: more numerically stable |
| Add "has capital gain/loss" columns | 0.9179 | Skipped: bins cover this |
| Add weight decay | 0.9174 | Skipped: negligible gain |
| Add a learning-rate scheduler | 0.9171 | Skipped: no real effect |
| Apply `log1p` to capital columns | 0.9112 | Rejected: made results worse |
| **Bins + low dropout + `Unknown` + logits** | **0.9210** | **Final setup** (the scheduler in this run had no effect) |

The most useful lessons were:

- **Test changes one at a time.** `log1p` looked like a sensible default, but it hurt, and I only found that by isolating it from a bundle of other changes.
- **A shallow copy can silently break early stopping.** `state_dict().copy()` still points at the live weights, so my original code never really restored the best model. `copy.deepcopy` fixed it.
- **Accuracy can hide a weak model.** Always guessing `<=50K` scores 76%, so I now also report recall, F1 and ROC-AUC.

To repeat the experiments, see [`experiments/`](experiments). `compare_versions.py` compares the original approach with the fixes, and `feature_study.py` produces the table above. Saved logs are in `experiments/results/`.

## Run it

### Option 1: Colab

Click the **Open in Colab** badge at the top, then choose **Runtime > Run all**. The data downloads automatically.

### Option 2: Locally

This was tested with Python 3.12, and training takes a few minutes on a CPU.

```bash
git clone https://github.com/ElsieElsevier/pytorch-income-classification.git
cd pytorch-income-classification
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook IncomeBracketStoppage.ipynb
```

### Make predictions

After running the notebook, use `predict_income` in a new cell. A trained model is also included at `model/income_classifier.pt`.

```python
predict_income([{
    'age': 45, 'workclass': 'Private', 'education_num': 16,
    'marital_status': 'Married-civ-spouse', 'occupation': 'Exec-managerial',
    'relationship': 'Husband', 'race': 'White', 'gender': 'Male',
    'capital_gain': 15000, 'capital_loss': 0, 'hours_per_week': 50,
    'native_country': 'United-States',
}])
```

## How it works

- **Data:** 32,561 training rows and 16,281 test rows, with about 24% earning over $50K. I obtained it through UMSL, and its columns and row counts match the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult), drawn from the 1994 US Census.
- **Cleaning:** removes a stray test row and whitespace, normalizes labels, and raises an error on unknown labels. It drops `education` (duplicated by `education_num`) and `fnlwgt` (a sampling weight).
- **Features:** 132 in total: standardized numeric values, one-hot categories, and one-hot quantile bins. All statistics are learned from the training portion only.
- **Model:** three hidden layers (128, 64, 32), each with batch normalization, ReLU and 10% dropout, producing one output score.
- **Training:** Adam (learning rate 0.001), batch size 64, and `BCEWithLogitsLoss`, with early stopping that restores the best weights. The split is 80/20 and stratified.

## Project structure

```text
IncomeBracketStoppage.ipynb   Full workflow with saved results
training_data.csv             Training data
test_data.csv                 Test data
model/                        Trained model and preprocessing statistics
experiments/                  Scripts and logs behind the comparison tables
requirements.txt              Python dependencies
```

## Limitations

- **The data is old.** The $50K line and the patterns behind it date from 1994.
- **It reflects historical inequality.** The training data shows large income gaps by sex and race, both of which are model inputs. Recall also varies by group: 0.65 for men and 0.55 for women on the final model. It shouldn't be used for real decisions about people, such as hiring or lending.
- **Tuning is limited.** I compared a few settings on validation data, but didn't run an exhaustive search.
- **The test score may be very slightly optimistic.** Early test results prompted the feature study, although the final setup was chosen on validation data.

## Acknowledgements

Built with [PyTorch](https://pytorch.org/). I used Claude as an AI assistant while writing and later revising this project.

## Contact

For questions, email jin.ken.bai@gmail.com.
