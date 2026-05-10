# CA-11 Digital Acquisition & FEC Contributions Audit

This repository contains the raw data, regression scripts, and analytical summaries for an independent audit of the Saikat Chakrabarti congressional campaign (CA-11). The analysis cross-references federal campaign finance filings (FEC) with the Meta Ad Library API to measure the causal relationship between digital ad spend and out-of-state donor acquisition.

The audit demonstrates how targeted digital acquisition schemes are utilized to subsidize national donor metrics, isolating an approximate $200 Meta ad acquisition cost per out-of-state donor (during the analyzed time periods).

## Repository Contents

### Raw Data Sets
* `meta_ad_library.csv`: Raw extraction of all Meta Ad Library entries for "Saikat for Congress" without date or active status filters. Includes fields for `ad_delivery_start_time`, `spend_lower_bound`, and `delivery_by_region` JSON strings.
* `fec_contributions_cross_ref.csv`: Itemized FEC disbursements and individual contributions corresponding to the Meta API time periods. Explicitly filtered for individual donors (`entity_type = IND`) to exclude organizational or aggregate ActBlue transfers.
* `voters_by_age_by_state.csv`: Census and demographic mapping of total citizen populations (in 100,000s) across U.S. states. Utilized to establish the Total Addressable Market (TAM) for per-capita baseline calculations and to control for geographic state size.

### Code
* `continuous_regression_analysis.py`: The primary Python script utilized to parse the raw CSVs, extract the geotemporal micro-windows, and execute the Continuous Difference-in-Differences (DiD) Dose-Response regression.

### Documentation (PDFs)
* **CA-11 Meta Ad Spend & FEC Contributions Audit (Updated).pdf**: The Executive Summary outlining the core findings. It breaks down the mechanical yield of the campaign's digital acquisition through isolated case studies. *Note: Example C (Dec 5 - Dec 22) clearly illustrates the pattern, showing how capping California ad delivery at 20% mathematically triggered a 3.03x spike in out-of-state daily donations, compared to a 74% increase in the local CA-only control baseline.*
* **Methodology Continuous DiD Dose Response For Campaign (Meta) Ad Spend.pdf**: The rigorous technical methodology detailing the statistical pipeline, variables, and baseline controls.

## Methodology Overview: Continuous DiD Dose-Response

While the Executive Summary provides isolated case studies, the underlying analysis utilizes a rigorous Continuous Difference-in-Differences Dose-Response model to strip away background noise (organic temporal momentum and fundraising emails) and prove statistical causality. 

Rather than viewing ad campaigns as continuous, the script extracts isolated geotemporal "micro-windows"—periods where localized out-of-state ad spend spikes above a threshold—and creates symmetrical Baseline Periods (the exact same duration immediately preceding the active ad buy) to create a clean "before-and-after" control.

The script runs the following DiD equation:

> Yield = β0 + β1(Active) + β2(Dose) + β3(Active × Dose) + γt + ε

**Variable Breakdown:**
* **β1 (Time Shock / `is_active`)**: Captures the baseline shift in donations due simply to time passing (e.g., natural momentum building closer to an election).
* **β2 (`dose_per_capita`)**: Controls for geographic selection bias (e.g., the campaign naturally spending more in states that already have a higher baseline of political engagement).
* **β3 (Interaction Term / `is_active:dose_per_capita`)**: The primary variable of interest. It represents the true isolated Return on Ad Spend (ROAS)—the marginal increase in donor yield directly caused by applying a financial dose, stripped of organic temporal momentum.

The model incorporates fixed effects to normalize time acceleration across different months and clusters standard errors by state to correct for autocorrelation.

## Execution

To run the analysis locally, clone the repository and execute the Python script:

```bash
python continuous_regression_analysis.py
```

*Note that `pandas`, `scipy.stats`, `matplotlib.pyplot`, and `statsmodels.formula.api` packages are needed.*

The script will output the regression summary (including p-values and R² scores) and generate a 2D scatter plot mapping the Empirical Financial Dose against the First Difference (Δ Yield).
