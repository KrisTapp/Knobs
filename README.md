# Knobs

This is the analysis code linked with the paper "Parameter Effects for ReCom Ensembles."

## Setup

To use this analysis code, you have to download the scores and by-district aggregates
and you have to clone the GitHub repository.

### Downloading the Artifacts

First download the scores, by-district aggregates, and helper code from 
[figshare](https://figshare.com/s/8d38a880772e5a7aa46e).
Choose the "Download all" option to get one large zip file that includes all zip files.
for the states and chambers along with all the scores integrated into a single Pandas
DataFrame, and helper code.

Once the download is complete, unzip the file. You can then delete the now redundant
original zip file. Also unzip the `scores-and-helpers.zip` file and delete the original.

### Cloning the Repository

Then clone and configure the GitHub repository. To do that, be in the directory where 
you want the repository and run this command:

```bash
git clone https://github.com/KrisTapp/Knobs
cd Knobs
```

Then, if you use virtual environments, create and activate one. 

Next, install the dependencies:

```bash
```pip install -r requirements.txt
```

Finally, create a `config.json` file in the root directory of the repository 
that specifies the paths to the scores and zipped directory from the first step above.

```json
{
    "scores-path": "/path/to/scores.parquet",
    "zip-dir": "/path/to/zipped"
}
```

## Usage
"Correlations.ipynb": This Notebook constains code that creates Table 1 of the paper, which reports the correlations between pairs of scores.

"Autocorrelations and redundancy.ipynb": This notebook contains the code for:
- Comparing the base ensembles to each other (section 5.1).
- Measureing the autocorrelation and effective samples size of the other ensembles (section 5.2).
- Comparing the level of redundancy of the ensembles (section 5.3), which relies on two files that were created elsewhere: redundancy_for_reversible.jsonl and redundancy_for_rest.jsonl.

"Compare ensembles.ipynb": This notebook generates tables and graphs from section 6 of the paper, which compares the ensembles.

"ensemble orders.ipynb": This notebook contains code to generate Figure 5 of the paper, which lines up the ensembles according to specific scores.

## Questions

Email questions to [Kris Tapp](mailto:ktapp@sju.edu?subject=Knobs question).


