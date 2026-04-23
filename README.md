# aou_phewas_plot

Publication-ready Manhattan plots for PheWAS results from the
[All of Us (AoU) Researcher Workbench](https://www.researchallofus.org/).

This repo is a thin plotting layer on top of
[PheTK](https://github.com/nhgritctran/PheTK): you run PheWAS with PheTK,
then feed the resulting table into the `Plot` class here. The output is a
publication-quality PDF.

## Usage on All of Us

All three steps below are meant to be run as cells in an AoU Jupyter
notebook.

### 1. Run PheWAS with PheTK

See the
[PheTK PheWAS module docs](https://github.com/nhgritctran/PheTK/blob/main/docs/phewas-module.md)
for the full set of options. A minimal call:

```python
!pip install PheTK --upgrade

from phetk.phewas import PheWAS

phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="phecode_counts.csv",
    cohort_file_path="cohort.csv",
    sex_at_birth_col="sex_at_birth",
    covariate_cols=["age", "sex_at_birth", "pc1", "pc2", "pc3"],
    independent_variable_of_interest="genotype",
    min_cases=50,
    min_phecode_count=2,
    output_file_path="phewas_results",   # PheTK always appends .tsv
)
phewas.run()
```

PheTK writes a **tab-separated** file (`phewas_results.tsv`). `Plot`
reads comma-separated files, so convert it once:

```python
import polars as pl
pl.read_csv("phewas_results.tsv", separator="\t") \
  .write_csv("phewas_results.csv")
```

The resulting CSV has the columns `Plot` expects: `phecode`,
`phecode_string`, `phecode_category`, `beta`, `p_value`,
`neg_log_p_value`, `converged`, ...

### 2. Clone this repo into your workspace

Run once per AoU workspace:

```python
!git clone https://github.com/yoshiji-lab/aou_phewas_plot.git
!pip install adjustText  # the only dependency not preinstalled on AoU
```

### 3. Import `Plot` and draw the Manhattan plot

```python
import sys
sys.path.insert(0, "aou_phewas_plot")   # make phewas_plot.py importable

from phewas_plot import Plot, set_global_font_size

set_global_font_size(14)

plot = Plot(
    phewas_result_csv_path="phewas_results.csv",
    phecode_version="X",
)
plot.manhattan(
    label_values="p_value",          # label the top hits by p-value
    label_count=15,
    fig_width=12,
    fig_height=4,
    axis_text_size=14,
    title_text_size=18,
    label_size=10,
    output_file_name="my_phewas_manhattan.pdf",
)
```

The PDF is written to the current working directory. For the full list
of keyword arguments (custom colour palette, Bonferroni override,
filtering by `phecode_category`, marker sizing by beta, y-axis limit,
legend, etc.) see `Plot.manhattan` in [`phewas_plot.py`](phewas_plot.py)
and the worked example in [`phewas.ipynb`](phewas.ipynb).

## Requirements

- Python 3.9+
- `polars`, `matplotlib`, `numpy`, `adjustText`

`polars`, `matplotlib`, and `numpy` are preinstalled on the AoU
Researcher Workbench; only `adjustText` needs `pip install`.

## Files

- `phewas_plot.py` — the `Plot` class (Manhattan plot, helpers, font setup).
- `phewas.ipynb` — worked example notebook.
