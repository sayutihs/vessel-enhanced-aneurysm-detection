# Vessel-Enhanced Deep Learning for Intracranial Aneurysm Detection

A two-stage 3D deep learning pipeline that detects and segments unruptured intracranial aneurysms in TOF-MRA brain scans. Instead of relying on a fixed vessel filter, the network **learns** its own vessel enhancement and uses it to guide detection.

**Final Year Project** — B.Eng. Electrical & Electronics Engineering, International Islamic University Malaysia (IIUM), 2026.

---

## Result

| Metric | Score |
|---|---|
| **Sensitivity** | **82%** |
| Dice | _(fill in)_ |
| Specificity | _(fill in)_ |

Evaluated on a held-out test set of unseen patients from the ADAM challenge dataset.

---

## The idea

Aneurysms are tiny — often just a few millimetres — and they sit on blood vessels that occupy a small fraction of the scan volume. A model looking at the raw scan has to find a needle in a haystack.

Most published approaches address this by pre-processing the scan with a **fixed** vessel filter (Hessian-based, Frangi, etc.) to highlight vasculature before segmentation.

This project takes a different approach: a small U-Net **learns** the vessel enhancement as part of training, and both stages are optimised together end-to-end.

```
Raw TOF-MRA volume  [B, 1, D, H, W]
        │
        ▼
┌─────────────────────────────┐
│  Vessel Enhancement Subnet  │   lightweight 3D U-Net (16→32→64)
│  → vesselness map (sigmoid) │
└─────────────────────────────┘
        │
        ▼  concatenate along channel dim
   [B, 2, D, H, W]   (raw volume + vesselness map)
        │
        ▼
┌─────────────────────────────┐
│  Aneurysm Detection Subnet  │   deeper 3D U-Net (32→64→128→256)
│  → aneurysm logits          │
└─────────────────────────────┘
```

Both subnets are trained jointly — the vessel subnet receives gradients from the detection loss, so it learns to highlight whatever the detector actually finds useful, rather than what a hand-designed filter assumes is useful.

---

## Dataset

[ADAM challenge](https://adam.isi.uu.nl/) (Aneurysm Detection And segMentation, MICCAI 2020) — 113 TOF-MRA volumes with voxel-wise aneurysm annotations.

Split **patient-wise** into 70% train / 15% validation / 15% test, with a fixed random seed for reproducibility. Splitting by patient (not by patch) prevents leakage of the same patient's anatomy across splits.

Expected folder structure:

```
dataset_root/
├── 10001/
│   ├── pre/TOF.nii.gz      # scan
│   └── aneurysms.nii.gz    # mask
├── 10002/
└── ...
```

---

## Key implementation decisions

**Aneurysm-biased patch sampling.** Aneurysm voxels make up a vanishingly small share of each volume, so uniformly random 3D patches would almost always be empty. Patches are sampled centred on a real aneurysm voxel 70% of the time and uniformly at random 30% of the time — enough positive signal to learn from, while still seeing normal anatomy.

**Tversky loss tuned for sensitivity.** The default loss is a combined Tversky + BCE with `alpha=0.3, beta=0.7`. Setting β > α penalises false negatives more heavily than false positives. In a screening context a missed aneurysm is far more costly than a false alarm a radiologist can dismiss, so the loss is deliberately biased toward recall. A standard Dice + BCE loss is also implemented for comparison.

**NIfTI caching to local SSD.** Decoding compressed `.nii.gz` volumes on every epoch is slow. Volumes are normalised once (z-score) and cached as raw `.npy`, then read back memory-mapped so each patch read touches only the sub-volume it needs.

**3D augmentation.** Random flips along all three axes and random 90° rotations in randomly chosen planes — cheap, label-preserving, and appropriate for volumetric data with no canonical orientation.

**Mixed precision + AMP.** FP16 autocast with gradient scaling to fit larger patches into GPU memory and speed up training.

**Dynamic padding.** Scans thinner than the patch size along any axis are zero-padded on the fly, so the loader never crashes on smaller volumes.

---

## Repository structure

```
├── model.py            # network architecture (both subnets + end-to-end wrapper)
├── dataset.py          # NIfTI loading, caching, patch sampling, augmentation, splits
├── train.py            # loss functions, metrics, trainer loop
└── colab_notebook.ipynb  # end-to-end pipeline: setup → train → evaluate → visualise
```

---

## Running it

The notebook is built for Google Colab with a GPU runtime.

```bash
pip install torch nibabel numpy scipy matplotlib optuna
```

1. Upload the ADAM dataset (zipped) to Google Drive.
2. Open `colab_notebook.ipynb` in Colab, select a GPU runtime.
3. Set `ZIP_PATH` and `DATASET_ROOT` to your paths.
4. Run the cells in order.

Quick check that the architecture builds and shapes line up:

```bash
python model.py
```

The notebook includes sanity checks before training — scan/mask dimension agreement, non-empty masks, NaN/Inf detection, and normalisation bounds — plus loss/metric curves, held-out test evaluation, and side-by-side visualisation of the raw scan, learned vesselness map, ground truth, and prediction.

Optional Optuna hyperparameter search over learning rate and loss weighting is included in the final cell.

---

## Built with

Python · PyTorch · NumPy · nibabel · matplotlib · Optuna

---

## Notes and limitations

- 113 volumes is a small dataset for 3D deep learning; results would benefit from external validation on an independent cohort.
- Evaluation is voxel-wise on patches. Full-volume, patient-level detection metrics (sensitivity per aneurysm, false positives per case) would be the more clinically meaningful measure.
- Trained on a single-institution challenge dataset — generalisation across scanners and acquisition protocols is untested.

---

## Author

**Ahmad Sayuti bin Hassan Shaari**
B.Eng. Electrical & Electronics Engineering (Hons), IIUM
[LinkedIn](https://www.linkedin.com/in/sayutihs/)
