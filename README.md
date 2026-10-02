# Titanic: Machine Learning from Disaster

A random-forest solution to the Kaggle [Titanic](https://www.kaggle.com/competitions/titanic) competition: predict which passengers survived the sinking, using passenger data such as class, sex, age, family and ticket information.

| | |
|---|---|
| **Kaggle username** | `<your-kaggle-username>` |
| **Best leaderboard score** | **`0.80143`** (accuracy on the public test set) |
| **Model** | Random Forest (shallow), selected by repeated cross-validation |

---

## Results

The score below was obtained with **the exact pipeline in `main.ipynb`**, run from top to bottom, with no manual changes between running the notebook and submitting `gender_submission.csv`.

| Version | Pipeline | Kaggle test score |
|---|---|---|
| v1 | Label-encoded binned features, deep tuned random forest | 0.75598 |
| **v2 (final)** | Engineered features + group-survival features + shallow random forest | **0.80** |

For reference, the trivial "all women survive, all men die" baseline scores about 0.766 on the leaderboard. The first version was below it, which signalled overfitting rather than real signal.

Cross-validation (repeated stratified 5-fold on the 891 training rows) is used for all model selection. A single 80/20 split was dropped because with only 179 validation rows it is accurate to about ±3 percentage points, too noisy to compare settings.

---

## Project structure

```
.
├── data/
│   ├── train.csv          # Kaggle training data (891 rows)
│   ├── test.csv           # Kaggle test data (418 rows)
│   └── gender_submission.csv     # generated predictions
├── notebooks/
│   └── main.ipynb         # full pipeline
└── README.md
```

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook notebooks/main.ipynb
```

Run all cells from top to bottom. The last cell writes `data/submission.csv`, ready to upload to Kaggle. All random seeds are fixed, so results are reproducible.

---

## Feature engineering

Train and test sets are combined for preprocessing, so imputation statistics and group sizes are computed over all 1,309 passengers. Only the training labels are ever used as targets.

### 1. Titles from names
The title is extracted from `Name` (`Mr`, `Mrs`, `Miss`, `Master`, ...). Rare titles are grouped into broader categories: `Military`, `Professional`, `Nobility`. `Mlle` and `Ms` become `Miss`, and `Mme` becomes `Mrs`. The title captures sex, age group and social status in one feature.

### 2. Missing values
| Column | Strategy |
|---|---|
| `Age` | Median age of the passenger's (`Title`, `Pclass`) group, with fallbacks to title median, then overall median. A `Master` is a boy and a `Mrs` is an adult, so this beats filling by class alone. |
| `Fare` | Median fare of the passenger's (`Pclass`, `Embarked`) group |
| `Embarked` | Filled with `C`, based on the class and fare of the two missing passengers |
| `Cabin` | Reduced to the deck letter. Missing values become `M`, and the single `T` deck is merged into `M` |

### 3. Ticket and fare
- **`TicketGroupSize`**: how many passengers share the same ticket.
- **`FarePerTicket`**: `Fare / TicketGroupSize`. The raw fare is the price of the whole group's ticket, so this is closer to what one person paid.

### 4. Family and simple indicators
- **`FamilySize`** = `SibSp + Parch + 1`
- **`IsAlone`**: `FamilySize == 1`
- **`HasCabin`**: whether a cabin was recorded
- **`IsChild`**: `Age <= 12`
- **`Sex_male`**: binary sex

`Age` and `FarePerTicket` are kept as continuous values. An earlier version binned them and label-encoded the bins in alphabetical order (for example `'100+'` sorted before `'20-40'`), which gave tree models meaningless orderings and threw information away.

### 5. Group survival features (largest single gain)
Families and travelling groups tended to live or die together. For each passenger, the pipeline finds the **women and children** (`female`, or title `Master`) who share their **ticket**, or who share their **surname and class**, and computes:

- the survival rate of those relatives,
- whether any relatives exist,
- how many there are.

To avoid label leakage:
- only **training labels** are used,
- a passenger is **never counted as their own relative**,
- inside cross-validation these features are **recomputed per fold** from that fold's training rows only,
- for test passengers they are computed from all training labels.

In cross-validation this feature lifted accuracy by roughly 1.5 points on top of the other engineered features.

### 6. Encoding
`Embarked`, `Title` and `Cabin` are one-hot encoded. No label encoding and no scaling are used, since tree models don't need them.

---

## Modelling

Six random-forest configurations are compared with repeated stratified 5-fold CV (5 repeats): the default forest and shallow variants (`max_depth` 5-8, `min_samples_leaf` 3-5, optional subsampling). The best configuration by mean CV accuracy is chosen automatically.

- **Selected model:** `<name printed as "Best:" in the notebook>`
- The selected model is **refit on all 891 training rows** before predicting the test set.

Shallow trees generalised better than the deeply tuned forest from the first version. The dataset is small and noisy, so a deep forest mostly memorises quirks of the training passengers.

---

## Lessons learned

- On a dataset this small, a single validation split is too noisy to guide tuning. Use repeated cross-validation.
- Feature quality mattered more than hyperparameter search. Handling encodings correctly, imputing ages by title and adding group survival moved the score more than any tuning.
- The leaderboard test set is a little harder than cross-validation suggests. The sex-only baseline scores about 78.7% in CV but 76.6% on the leaderboard, so expect CV to run roughly 2 points optimistic.
- Any feature built from target labels must be computed out-of-fold, excluding the row itself.