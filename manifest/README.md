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
