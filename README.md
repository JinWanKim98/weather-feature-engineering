# Rain in Australia — the front half of the pipeline

**Four-person group project (CSCI 316 Big Data Mining Techniques and Implementation).** The task was
an end-to-end scikit-learn project on the Rain in Australia dataset — 145,460 daily weather records
from 49 stations, predicting whether it rains tomorrow.

**My part: sections (a) and (b) — discover and visualise the data, and prepare it for the
algorithms.** That is cells 0 to 27 of the 46 in the notebook. Sections (c), (d) and (e) — training
the three models, tuning them and evaluating them — were done by other members of the group.

The front half is where the decisions are made that the back half cannot undo, so this README is
about four of them: which columns to throw away, where the train/test split goes, which new features
are worth building, and how to hand the tuning step something it can actually decide.

---

### 1. The column-dropping rule everybody uses is wrong on this dataset

Four columns are missing more than a third of their values, which is usually where a cleaning script
starts deleting. So before deleting anything I plotted how much of each column is missing against how
strongly it correlates with the target.

![Missing rate versus predictive power](images/missing_rate_vs_predictive_power.png)

The three columns furthest to the right are also near the top.

| Column | Missing | \|Correlation with target\| | Rank among 16 numeric features | Kept? |
|---|---|---|---|---|
| `Sunshine` | 48.0% | 0.451 | **1st** | **kept** |
| `Cloud3pm` | 40.8% | 0.382 | **3rd** | **kept** |
| `Cloud9am` | 38.4% | 0.317 | 4th | **kept** |
| `Evaporation` | 43.2% | 0.119 | 12th | dropped |

Dropping on a missing-rate threshold would have deleted the first, third and fourth most informative
features in the dataset. `Evaporation` is the only column that is bad on both axes, so it is the only
one that goes. The other three are imputed inside the pipeline instead.

The same plot answers a second question. Row-wise deletion is the other reflex, and the notebook
computes what it would cost rather than guessing: of the 142,193 rows with a known target, only
**58,278 — 41% — have no missing value anywhere.** A `dropna()` would throw away **83,915 rows** to
avoid imputing. That number is why imputation is not a compromise here; it is the only option that
keeps the dataset.

---

### 2. A feature that carries more signal than either column it is made from

`MaxTemp` correlates with the target at −0.160. `MinTemp` at +0.083. Their difference correlates at
**−0.337** — more than twice either source.

![TempRange against its two source columns](images/temprange_vs_sources.png)

A wide gap between the day's high and low is a dry, clear day; a narrow one is overcast. Neither
column says that on its own, because both of them move with the season and the latitude as well.
The difference cancels what they share and leaves what they don't.

I tested five candidates on the training set only and kept two:

| Candidate | Definition | Correlation | Verdict |
|---|---|---|---|
| **`TempRange`** | `MaxTemp − MinTemp` | **−0.337** | keep |
| **`HumidityChange`** | `Humidity3pm − Humidity9am` | **+0.268** | keep |
| `PressureChange` | `Pressure3pm − Pressure9am` | +0.080 | reject — weak |
| `CloudChange` | `Cloud3pm − Cloud9am` | +0.047 | reject — weak |
| `WindChange` | `WindSpeed3pm − WindSpeed9am` | −0.005 | reject — no signal |

**The "beats its inputs" result belongs to `TempRange` alone, and it is worth being exact about
that.** `HumidityChange` is +0.268 while `Humidity3pm` on its own is +0.445 — the derived feature is
weaker than the column it came from. It earns its place by describing the direction humidity moved
during the day, which is a different fact, not a stronger version of the same one. Three of five
candidates were rejected outright. Showing only the two that worked would make the hit rate look
like luck rather than a test.

---

### 3. The feature I did not build

A seasonal feature out of the `Date` column is the obvious next idea, and it does not work here.
Monthly rain probability moves only between 19.3% and 26.9% across the year, and the correlation is
**0.007** — flat.

The reason is in the data: the 49 stations span tropical sites that get their rain in summer and
temperate sites that get it in winter. Pooled into one column, the two patterns cancel. A per-region
seasonal feature might survive, but that is a different feature from the one I tested, so no seasonal
feature was added and the notebook says why rather than leaving a gap.

The three figures in this section appear in the notebook's markdown without a computed cell, so I
checked them against the source CSV. They hold. The test behind them used the month as a plain
number, which treats December and January as eleven apart when they are adjacent — a cyclical
encoding is a different test, and I ran neither. Section 3 rules out the feature I tried, not
seasonality.

---

### 4. Where the split goes

Cleaning steps are split by whether they need to look at the data:

| Before the split | Inside the pipeline, after the split |
|---|---|
| Drop rows with no target | Impute missing values |
| Drop the `Evaporation` column | Scale numeric features |
| | One-hot encode categoricals |

Dropping a row or a column uses no information from the data, so it is safe to do up front. An
imputation does: filling a gap with the median means computing that median, and computing it over
the whole dataset before splitting leaks the test set into the training set and makes the final
score optimistic. So every step that learns a number lives inside a `Pipeline` that is fitted on the
training data alone.

The split is stratified because the target is imbalanced — 77.58% No, 22.42% Yes. After splitting,
the full set, the training set and the test set all read 77.58 / 22.42. Without stratification the
test set's balance drifts and the evaluation measures the drift as well as the model.

---

### 5. The switch, and what the search decided about it

The specification asked for a user-defined transformer with a parameter that turns the new features
on and off, tunable in step (d). That is a small requirement with a consequence that is easy to miss:

**with `add_features=True` the matrix is 116 columns, with `add_features=False` it is 114.** Any
`ColumnTransformer` built from a hard-coded column list works in one setting and breaks in the other
— which means the hyperparameter cannot be searched at all. So the columns are selected by dtype
with `make_column_selector`, and the pipeline is valid in both positions of the switch:

```python
class WeatherFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_features=True):
        self.add_features = add_features        # stored unchanged, so clone() works
    def fit(self, X, y=None):
        return self                             # nothing is learned; the features are row-wise
    def transform(self, X):
        X = X.drop(columns=['Date'])
        if self.add_features:
            X['TempRange'] = X['MaxTemp'] - X['MinTemp']
            X['HumidityChange'] = X['Humidity3pm'] - X['Humidity9am']
        return X
```

Then the grid search was free to switch the features off if they did not help.

**The grid-search output shows `add_features: True` for all three models** — logistic regression,
random forest and gradient boosting. Three independent searches, each optimising its own model, each
keeping the two engineered columns.

That is the part I would point at. Anyone can add a column and assert it helps. Putting it behind a
switch and handing the switch to the tuner means the claim is settled by the search rather than by
me, and it came back the same way three times.

---

### 6. Reading the notebook

**The grid-search output shows `add_features: True` for all three models. A summary paragraph
further down reads it as logistic regression only; the cell output is the record.**

The notebook is committed exactly as it was submitted, apart from the student ID numbers described
below, so both the output and the paragraph are here to read. Where they disagree, the executed cell
is what happened.

---

### Repository Structure

```
weather-feature-engineering/
├── notebook/
│   └── weather_task1_sklearn.ipynb   # 46 cells, outputs saved. My part is cells 0-27.
├── images/                           # two charts, extracted from the notebook's own outputs
└── README.md
```

### How to Run

The dataset is not committed — it is 14 MB and publicly available:

```bash
# https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
# download weatherAUS.csv into notebook/
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook notebook/weather_task1_sklearn.ipynb
```

Every cell output is saved in the committed file, so the notebook reads correctly without the CSV;
the download is only needed to re-run it.

### Provenance

The notebook is the group's submitted file. My sections are (a) and (b), cells 0 to 27 — the
exploration, the cleaning decisions, the feature engineering and the preprocessing pipeline. Cells 28
to 45 — model selection, training, tuning and evaluation — are the work of other group members, and
the results quoted in section 5 above are theirs.

**Teammates are named in the notebook.** The four university student ID numbers that sat next to
those names have been removed: they identify a person outside this project and carry no credit.
Nothing else in the notebook has been altered.

A second task on the same dataset, implemented in Spark MLlib, was part of the same assignment. I did
not write it and it is not in this repository.

### Limitations

- **The engineered features were chosen by correlation with the target, computed on the training
  set.** Correlation is linear; a feature with a real non-linear relationship would score near zero
  and be rejected by this screen. `WindChange` at −0.005 may be a genuine null or may be a limit of
  the test used.
- **`TempRange` and `HumidityChange` are not independent of the columns they came from**, which are
  still in the matrix. The search kept the pair anyway, but this is not a clean measurement of how
  much the engineered features add on their own.
- **The seasonal test was run pooled across all 49 stations.** The cancellation argument in section 3
  predicts that a per-station or per-climate-zone version would behave differently. I did not build
  one, so that remains a prediction rather than a result.
- **Imputing a column that is half missing puts the median into roughly half its rows.** Keeping
  `Sunshine` was the right call on the evidence, but the version the models saw is a much flatter
  column than the one that was measured.
