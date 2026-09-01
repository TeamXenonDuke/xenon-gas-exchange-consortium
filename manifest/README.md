# Manifest batch processing

`batch_pipeline.py` is a first-version batch runner for the xenon gas exchange
pipeline. It replaces manually authored Python configs with a CSV manifest
containing one subject per row.

## Quick start

Run a dry-run preflight check. It validates inputs and creates frozen JSON
config snapshots, but does not reconstruct images:

```bash
python manifest/batch_pipeline.py
```

Run the same manifest through the existing pipeline:

```bash
python manifest/batch_pipeline.py --run
```

Discover all subject folders under `data/` without manually adding CSV rows.
This dry run replaces `manifest_example.csv` with one editable row per valid
subject and writes each subject to `batch_status.csv` as `planned`. It uses
`manual_vent` segmentation and looks beside each Twix/MRD file for
`mask_reg_corrected.nii`, falling back to `mask_reg.nii`. It also copies the
first valid `rbcm` value (or fifth CSV column when that header is absent) from a
subject's `spectroscopy`/`Spectroscopy` folder, rounded to three decimals.
Missing masks or RBC:M values are left blank and logged for review. Discovered
subjects use Plummer reconstruction, with oscillation analysis and VC correction enabled:

The generated CSV also expands the active defaults for reconstruction, output
folder, segmentation, registration, bias-field correction, reference data,
ventilation normalization, and trajectory settings. This makes the generated
manifest directly reviewable and editable. Fields that require subject-specific
confirmation (`hb` and lung-volume correction) remain blank.

`recon_size`, `del_x`, `del_y`, `del_z`, `key_radius_pct`, and `n_skip_start`
are deliberately blank in discovered manifests so they inherit the enforced
defaults from `base_config.Config`. They may still be set as per-subject CSV
overrides when needed.

For Twix inputs, age, sex, height, and weight are also copied into the generated
CSV when available in the Dixon header. If the header does not provide a complete
set, those cells remain blank rather than receiving audit defaults. The existing
pipeline uses raw-input (or `.mat`) demographics when those CSV cells are blank.
Any nonblank CSV demographic value takes priority and is used during processing,
including age, sex, and height in VC correction; weight is retained in outputs
but is not an input to the current VC-correction calculation.

```bash
python manifest/batch_pipeline.py --discover
```

Use another data root when needed. The runner chooses `recon` when it finds
`.dat` or `.h5` input, and chooses `readin` only when the folder has `.mat`
input but no raw data:

```bash
python manifest/batch_pipeline.py --discover --data-root /path/to/data
```

For discovered raw-data subjects, gradient delays are inherited from
`base_config.Config`. Ramp time is read from the Dixon Twix header when possible;
otherwise the manifest records `90`.

To process the same automatically discovered subjects, add `--run`:

```bash
python manifest/batch_pipeline.py --discover --run
```

Use another manifest or data root when needed:

```bash
python manifest/batch_pipeline.py \
  --manifest manifest/my_batch.csv \
  --data-root /path/to/data \
  --run
```

## Manifest columns

Only `subject_id` and `process_mode` are required. With an empty `data_dir`,
the runner uses:

```text
data/<subject_id>
```

| Column                    | Purpose                                                             |
| ------------------------- | ------------------------------------------------------------------- |
| `subject_id`              | Subject identifier and generated-config filename.                   |
| `data_dir`                | Optional subject data directory.                                    |
| `process_mode`            | `recon` for raw `.dat`/`.h5`; `readin` for `.mat`.                  |
| `rbc_m_ratio`             | Optional manual RBC:M ratio.                                        |
| `hb`                      | Optional hemoglobin. Enables Hb correction if supplied.             |
| `corrected_lung_volume`   | Optional target lung volume. Enables volume correction if supplied. |
| `recon_proton`            | `true` or `false`.                                                  |
| `recon_key`               | Normally `robertson` or `plummer`.                                  |
| `scan_type`               | Optional scanner sequence label.                                    |
| `del_x`, `del_y`, `del_z` | Optional gradient-delay overrides in microseconds.                  |
| `oscillation_analysis`    | Enables the existing RBC oscillation branch.                        |
| `output_folder`           | Subject-local pipeline output folder.                               |
| `combine_reports`         | Optional: merge gas exchange, discovered spectroscopy, and oscillation PDFs. |

### Extended controls and demographics

The manifest also supports per-subject overrides for `vc_correction`,
`segmentation_key`, `manual_seg_filepath`, `registration_key`,
`manual_reg_filepath`, `bias_key`, `reference_data_key`, `bag_volume`,
`vent_normalization_method`, `n_skip_start`, `n_skip_end`, `traj_type`,
`traj_scaling_factor`, `dicom_proton_dir`, and `multi_echo`. Leave an optional
cell blank to retain the value in `base_config.Config`.

`age`, `sex`, `height_cm`, and `weight_kg` are recorded in the generated JSON
snapshot. Provided manifest values take priority. For Twix inputs with blank
demographic cells, the runner reads the Dixon Twix header; if that is not
available, it records documented defaults of 50 years, `M`, 170 cm, and
70 kg. These values are audit metadata: the existing Python pipeline continues
to obtain patient information from the Twix/MRD header during processing.

Set `combine_reports` to `true` to write
`<subject_id>_combined_report.pdf` in the subject directory. The pipeline
merges the gas-exchange report from `output_folder` (normally `gx_batch`), the
best matching PDF found in another subject subfolder, and the oscillation
report when present. If no spectroscopy PDF is found, that section is skipped.

When `rbc_m_ratio` is blank, the pipeline attempts to calculate it from static
spectroscopy. If that calculation cannot be completed, it uses `0.455` and
emits a warning. A supplied manifest value always takes priority.

## Generated files

```text
manifest/
├── batch_pipeline.py
├── manifest_example.csv
├── generated_configs/
│   └── <subject_id>_config.json
└── batch_status.csv
```

`batch_status.csv` is updated after each subject. It records validation,
planned, completed, failed, and needs-review states, together with data paths,
generated-config paths, output paths, timestamps, and error messages.

## Design

The runner directly constructs `config.base_config.Config` and then calls one
of the existing project functions:

- `main.gx_mapping_reconstruction`
- `main.gx_mapping_readin`

It does not replace reconstruction, segmentation, registration, statistics,
NIfTI/MAT/CSV export, or PDF report generation.

Processing is serial because the existing pipeline uses a shared project-level
`tmp/` directory. Per-subject temporary directories are needed before safely
adding parallel workers.

## Inspiration

The execution approach comes from the existing Python
`script_process_batch.py`, which imports configs and calls the same processing
functions.

The manifest-driven config-generation approach comes from the MATLAB
`write_config_2.m` workflow: subject-level values are assembled into a config,
preserved for audit, then processed. This version stores metadata in editable
CSV rows instead of hard-coded MATLAB paths and template replacement rules.
