# Manifest batch processing

This batch runner discovers subjects under one large data directory, writes one
editable CSV row per subject, and runs the existing pipeline without importing a
subject-specific Python config file.

## Recommended workflow

Generate the table first:

```bash
python manifest/batch_pipeline.py \
  --discover \
  --data-root /path/to/subjects \
  --manifest manifest/manifest.csv
```

When both MRD and Twix inputs exist, MRD is selected by default. Use
`--prefer twix` to select Twix instead. Discovery refuses to overwrite an
existing table unless `--overwrite` is supplied.

Review and edit `manifest/manifest.csv`, then validate it without processing:

```bash
python manifest/batch_pipeline.py --manifest manifest/manifest.csv
```

Run discovery and processing in the same environment. For example, a manifest
created with Windows `C:\...` paths is not portable to WSL; create it in WSL so
the table contains `/mnt/c/...` paths.

Run the validated rows serially:

```bash
python manifest/batch_pipeline.py --manifest manifest/manifest.csv --run
```

The two-step workflow is intentional. `--discover --run` requires the explicit
`--allow-unreviewed` acknowledgement, and rows with unresolved discovery or
validation errors still will not run.

## Supported layouts

The original layout remains supported:

```text
subjects/
├── subject-001/     # raw files may be directly inside this folder
└── subject-002/
```

The SOX layout is also supported:

```text
subjects/
└── SOX-002/
    ├── data/        # Twix files
    │   └── mrd/     # MRD files
    ├── spectroscopy/
    ├── gx/
    └── osc_img/
```

Discovery scans the data root once, finds subjects at nested depths, and records
the subject root, exact input directory, exact Dixon/MAT source file, and all
available input types. It also looks for:

- `mask_reg_corrected.nii`, then `mask_reg.nii`
- a spectroscopy CSV containing an `rbcm` column
- age, sex, height, weight, and ramp time in the selected Dixon header

If multiple Dixon files, masks, or distinct RBC:M values match, the row is
marked `needs_review` instead of silently choosing an ambiguous value. Edit the
row, set `discovery_status` to `ready`, and clear the resolved message before
running it.

For Twix rows, `del_x`, `del_y`, and `del_z` must be reviewed and populated.
Boolean cells accept only `true/false`, `yes/no`, or `1/0`.

Blank optional cells inherit `config/base_config.py` defaults. The generated
JSON snapshot records the actual config used for each subject.

## Output

Runtime audit files are written locally:

```text
manifest/generated_configs/<subject_id>_planned_config.json
manifest/batch_status.csv
```

The planned snapshot exists even when processing fails. A successful pipeline
run also writes `<subject_id>_config_gx_imaging.json` in the subject output
folder; both paths are recorded in `batch_status.csv`. One subject failure does
not prevent the remaining subjects from running.
