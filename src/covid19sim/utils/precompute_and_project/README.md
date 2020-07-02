# Data Wrangling

In aggregating data from various sources, a major challenge is to make sure that the underlying assumptions are respected when they are used together.
The scripts in this folder project the country level data to a specific location within that country.
For our use case, we have a country level data in `src/covid19sim/configs/simulation/country/canada.yaml` and we want to project the data to Montreal's demographical structure.

`demographics.py` projects the age breakdown at a country level to a region level.
`contacts.py` projects contact matrices from country level to a region level.

## How to run?

```python
python demographics.py --country ../../configs/simulation/country/canada.yaml --region ../../configs/simulation/region/montreal.yaml
```
Above will append to `montreal.yaml` precomputed and projected values with necessary comments to understand.
