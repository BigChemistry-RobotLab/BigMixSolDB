<h1 align="center">
    <b>BigMixSolDB: Extraction of a solubility database in solvent mixtures with an uncertainty-quantified large language model-based pipeline</b><br>
</h1>

We present a fully automated, large language model (LLM) based pipeline to extract complex tabular and textual chemical data directly from scientific PDFs. Based on a benchmark against existing large solubility databases ([BigSolDB 2.0](https://doi.org/10.1038/s41597-025-05559-8) and [MixtureSolDB](https://doi.org/10.1038/s41597-026-07047-z)), we have found that the presented workflow extracts data with accuracies comparable to the aleatoric uncertainty in these datasets.



# BigMixSolDB Extraction Workflow

This repository contains the workflow for extracting experimental solubility data from scientific papers.

The workflow covers the steps used in the article:

1. Parse PDFs with:
    - Docling to Markdown.
    - VLM to Markdown.
2. Extract structured solubility data from Markdown to YAML with an LLM.
3. Extract molecule names into an editable JSON file with empty SMILES placeholders.
4. Optionally resolve SMILES in that JSON with ChemScript and PubChem fallback.
5. Build a standardized, filtered CSV directly from YAML files, optionally filling SMILES from the previous JSON file and recording filtering diagnostics.
6. Curate the publication dataset by collapsing duplicate measurements reported by multiple papers and harmonizing solvent display names.
7. Compare the curated CSV against a reference CSV and write a JSON error report.

## Installation

Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Provider Configuration

Two provider families are supported:

- OpenAI-compatible endpoints
- Google Gemini

### OpenAI-compatible

Set these environment variables before running `convert_vlm.py` or `extract_yaml.py`:

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
```

Override them per command with `--api-base` and `--api-key` if needed.

### Gemini

Set:

```bash
export GEMINI_API_KEY="your-api-key"
```

## Directory Layout

- `scripts/`: command-line entry points
- `src/bigmixsoldb/`: reusable implementation code
- `prompts/`: default prompts used by the extraction pipeline (Note: LightOnOCR-2-1B does not require a prompt for page transcription.)

## Usage

### 1a. Parse PDF with Docling

```bash
python scripts/convert_docling.py papers/*.pdf --output-dir outputs/docling
```

This writes one Markdown file per PDF.

### 1b. Parse PDF with a VLM

OpenAI-compatible example:

```bash
python scripts/convert_vlm.py papers/*.pdf \
  --provider openai \
  --model lightonai/LightOnOCR-2-1B-bbox \
  --api-base "$OPENAI_API_BASE" \
  --api-key "$OPENAI_API_KEY" \
  --output-dir outputs/vlm
```

Gemini example:

```bash
python scripts/convert_vlm.py papers/*.pdf \
  --provider gemini \
  --model gemini-2.5-pro \
  --output-dir outputs/vlm
```

The script renders each PDF page to an image, sends it to the selected model, and concatenates the page Markdown into a single `.md` file.

#### Run a local vLLM server (OpenAI-compatible)

If you prefer to run an OpenAI-compatible vLLM server locally, install vLLM and its dependencies and start the server. The repository includes a small wrapper script at `scripts/start_vlm_server.sh` that runs `vllm serve` with sensible defaults (model, host, port, API key).

Quick install and start (Linux):

```bash
# Install PyTorch for your CUDA (see https://pytorch.org/get-started/locally)
pip install torch torchvision

# Install vLLM and helpers
pip install vllm transformers accelerate huggingface_hub

# Optional (improves GPU/8-bit perf): pip install bitsandbytes xformers
```

Start the server (defaults to `lightonai/LightOnOCR-2-1B-bbox` on port `8000`):

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
./scripts/start_vlm_server.sh
# Or pass additional vLLM args:
./scripts/start_vlm_server.sh --gpu-memory-utilization 0.8
```

Once the server is running, use the OpenAI-compatible example above with `--api-base "$OPENAI_API_BASE"` and `--api-key "$OPENAI_API_KEY"`.

### 2. Extract YAML from Markdown

```bash
python scripts/extract_yaml.py outputs/docling/*.md \
  --provider openai \
  --model gpt-5.1 \
  --api-base "$OPENAI_API_BASE" \
  --api-key "$OPENAI_API_KEY" \
  --prompt-file prompts/extract_yaml_prompt.txt \
  --output-dir outputs/yaml
```

### 3. Extract Molecule Names and Create SMILES Placeholders

```bash
python scripts/extract_molecule_names.py outputs/yaml --output molecules.json
```

This writes a JSON file with `solutes` and `solvents`, each containing editable `smiles` fields.

### 4. Optionally Resolve SMILES with ChemScript and PubChem

```bash
python scripts/resolve_molecules.py molecules.json --output molecules_resolved.json
```

If ChemScript is installed in the active Python environment, it is used first. If ChemScript is unavailable or fails for a name, the script falls back to PubChem and records lookup metadata in the JSON. ChemScript is optional and is not listed as a required dependency because it is proprietary software.

### 5. Build a Standardized Filtered CSV

Solubilities that require a solvent density are converted only from an offline,
exact-temperature manifest. Create or refresh that manifest separately:

```bash
python scripts/prefetch_densities.py outputs/yaml \
  --molecules molecules_resolved.json \
  --output outputs/density_manifest.json
```

The prefetch command seeds observations from `data/bigsoldb_densities.csv` when that
file exists and queries PubChem only for identities and temperatures not satisfied
locally. It retains alternative and rejected evidence for auditing. Dataset builds
never query PubChem and never interpolate or guess a density. Manifests are
schema-versioned, validated before use, and written atomically.

The command can scan either extracted YAML inputs or an existing CSV. Use
`--local-density-csv PATH` to select a different local density source, or add
`--no-pubchem` to create the manifest entirely offline from local observations while
recording any unsatisfied density requests.

```bash
python scripts/build_csv.py outputs/yaml \
  --molecules molecules_resolved.json \
  --density-manifest outputs/density_manifest.json \
  --conversion-report outputs/solubility_conversion_report.csv \
  --stats-output outputs/bigmixsoldb_filtering_stats.json \
  --output outputs/bigmixsoldb.csv
```

This script:

- standardizes all YAML records into the publication CSV schema
- merges them into one dataset
- converts supported single-solvent solubility units to fractions, with exact-temperature
  density evidence where required
- parses multiplier and power-of-ten notation with one shared, anchored unit grammar;
  ambiguous OCR forms remain unchanged and are recorded in the conversion audit
- applies completeness, invalid-result, reference, supported-solubility-unit, and
  condition-deduplication filters as separate sequential stages (the supported-unit
  filter runs before condition deduplication)
- prints build statistics such as total inputs found, per-system counts, and filtering removals by reason
- optionally writes reconciled stage-by-stage attrition as JSON with `--stats-output`
- writes a genuinely pre-filter, pre-condition-deduplication `_unfiltered.csv`
- optionally writes a row-level conversion audit with parsed notation, scale divisor,
  exponent normalization, basis, attempted result, solvent identity, and density evidence
  using `--conversion-report`

The final CSV contains the standardized schema:

- compound and solvent names
- optional SMILES columns
- concentration fields
- solubility, temperature, and pressure
- review flag
- DOI derived from the input filename

### 6. Collapse Duplicate Measurements Across Papers

```bash
python scripts/deduplicate_cross_doi.py outputs/bigmixsoldb.csv \
  --output outputs/bigmixsoldb_deduplicated.csv \
  --duplicate-report outputs/bigmixsoldb_duplicate_groups.csv \
  --doi-duplicate-report outputs/bigmixsoldb_duplicate_dois.csv
```

This publication-curation step identifies the same measurement reported under different DOIs using normalized chemical structures and conditions. By default, solvent/concentration pairs are order-independent, stereochemistry is ignored, missing pressure is treated as unspecified, recognized pressure values use a 2% relative tolerance, and compatible differences in reported numeric precision can be matched. The retained row combines DOI provenance and fills missing metadata from its duplicates. Ambiguous precision matches are reported but not merged.

Only cross-DOI duplicates are merged by default. Add `--same-doi-duplicates` when the input should also be deduplicated within individual papers. The two optional reports provide measurement-group and per-DOI audit trails.

### 7. Harmonize Solvent Display Names

```bash
python scripts/standardize_solvent_names.py outputs/bigmixsoldb_deduplicated.csv \
  --output outputs/bigmixsoldb_curated.csv \
  --audit-output outputs/bigmixsoldb_solvent_name_map.csv
```

This changes only the `Solvent 1`, `Solvent 2`, and `Solvent 3` display-name columns. Solvents are grouped by the InChIKey derived from their SMILES, then assigned a curated common name when one is available or the most frequent observed name otherwise. The audit CSV records every original-to-canonical mapping and its source. Structure and measurement columns are preserved.

Together, steps 6 and 7 turn the build output into the curated CSV used for downstream reference comparisons, figures, and publication metrics while retaining auditable reports of each normalization decision.

### 8. Compare Against a Reference Dataset

Compare against MixtureSolDB:

```bash
python scripts/compare_csv.py \
  --predicted outputs/bigmixsoldb.csv \
  --reference reference_code/data/MixtureSolDB.csv \
  --reference-format mixturesoldb \
  --output outputs/mixturesoldb_comparison.json
```

Compare against BigSolDB:

```bash
python scripts/compare_csv.py \
  --predicted outputs/bigmixsoldb.csv \
  --reference reference_code/data/bigsoldb.csv \
  --reference-format bigsoldb \
  --output outputs/bigsoldb_comparison.json
```

The comparison is conservative. It only compares single-solvent and binary-mixture records that can be normalized reliably.

## Additional Utilities

### Per-YAML Standardized CSVs

```bash
python scripts/postprocess_yaml.py outputs/yaml \
  --molecules molecules_resolved.json \
  --output-dir outputs/csv
```

Each YAML file becomes one standardized CSV. This is useful if you want per-paper intermediate outputs instead of a single merged dataset.

### Create a Fine-Tuning Dataset

Pair Markdown inputs with curated YAML labels that share the same filename stem and write them as chat-style JSONL for supervised fine-tuning.

```bash
python scripts/create_finetune_dataset.py outputs/docling \
  --labels data/train \
  --output data/ft_dataset_train.jsonl

python scripts/create_finetune_dataset.py outputs/docling \
  --labels data/val \
  --output data/ft_dataset_val.jsonl
```

You can point `inputs` at either `outputs/docling` or `outputs/vlm`, and `--labels` at any YAML files or directories. Matching is done by filename stem, so `10.1021_...md` pairs with `10.1021_....yml`.

### Fine-Tune a Hugging Face Model

Install PyTorch for your hardware first, then install the finetuning extras:

```bash
pip install torch
pip install -e ".[finetune]"
```

Run supervised fine-tuning with a train split and optional validation split:

```bash
python scripts/finetune.py \
  --model Qwen/Qwen3-8B \
  --train-data data/ft_dataset_train.jsonl \
  --validation-data data/ft_dataset_val.jsonl \
  --output-dir checkpoints
```

You can also pass a YAML or JSON config file to supply `SFTConfig` arguments such as batch size, learning rate, number of epochs, or mixed-precision settings.

# License
This repository is licensed under the AGPLv3 License. See [LICENSE](LICENSE) for details.

If you would like to use the code in this repository for commercial purposes, please contact the BigChemistry team at [info@bigchemistry.nl](mailto:info@bigchemistry.nl) to discuss licensing options.
