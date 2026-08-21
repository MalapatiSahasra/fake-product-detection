# Dataset Acquisition & Characteristics

## 1. Data Acquisition

The evaluation dataset is drawn from a real, publicly available image
corpus rather than a synthetically rendered one: the **"Fake Brand
Dataset (Adidas, Nike, Gucci, Puma, LV)"**, published on Kaggle by user
*prashanthgolla* under a Creative Commons Attribution-NonCommercial 4.0
(CC BY-NC 4.0) license. The raw corpus contains genuine and counterfeit
product photographs across five brands, sourced from e-commerce listings,
warehouse/seizure photography, and close-up product shots — a
heterogeneous, uncurated mix that reflects real marketplace imagery
rather than studio-controlled photography.

| Brand | Genuine (raw) | Counterfeit (raw) |
|---|---:|---:|
| Nike | 260 | 253 |
| Puma | 199 | 91 |
| Adidas | 178 | 140 |
| Gucci | 190 | 152 |
| LV (Louis Vuitton) | 28 | 74 |
| **Total** | **855** | **710** |

`sort_kaggle_dataset.py` sorts each brand's `*_genuine/` and `*_fake/`
folders into this project's layout and, critically, carves out a
**held-out reference gallery** from the genuine images before anything
else touches them.

## 2. Held-Out Reference Gallery

Because "genuine" here spans hundreds of distinct products, angles, and
photographic styles per brand rather than one fixed physical unit, the
logo and color modules compare each test photo against a *gallery* of
genuine reference images (Section 3 of the main paper) rather than a
single master image. A fixed, seeded fraction (8%) of the genuine images
per brand is reserved into `dataset/gallery/` and is **never** included
in the evaluation set:

| Brand | Gallery (held out) | Evaluation seed pool |
|---|---:|---:|
| Nike | 21 | 239 |
| Puma | 16 | 183 |
| Adidas | 14 | 164 |
| Gucci | 15 | 175 |
| LV | 2 | 26 |
| **Total** | **68** | **787** |

This separation is essential to reproducibility: without it, a genuine
test image could sit inside its own reference gallery and match itself,
silently inflating measured accuracy. All counterfeit images are
gallery-independent (the gallery is genuine-only) and pass through to
the seed pool unchanged: 710.

## 3. Dataset Composition

Because the raw seed pool (787 genuine, 710 counterfeit) already exceeds
what earlier, smaller-scale versions of this pipeline used, the
augmentation target was set to **800 images per class** — high enough
that the final set is overwhelmingly real photographs, with only a light
synthetic top-up to reach an exact, class-balanced round number.

| Class | Seed Images | Augmented Images | Total Images | Class Ratio |
|---|---:|---:|---:|---|
| Genuine (Authentic) | 787 | 13 | 800 | 1 : 1 |
| Counterfeit | 710 | 90 | 800 | 1 : 1 |
| **Total** | **1,497** | **103** | **1,600** | — |

97% of the final evaluation set (1,497 of 1,600 images) is unmodified
real photography; augmentation contributes only the minimal top-up
needed for an exact 1:1, round-number balance. This is a deliberate
inversion of the small-seed-corpus regime (where augmentation must do
most of the work) once a sufficiently large real corpus is available.

## 4. Augmentation Pipeline & Simulated Capture Conditions

Each seed image not retained as-is is expanded through three
independent, randomly parameterized perturbations applied in sequence,
chosen to approximate the dominant sources of variance in repeated
smartphone photography of a physical object:

| Perturbation | Method | Parameter Range | Simulates |
|---|---|---|---|
| Geometric rotation | Affine rotation about image center (`cv2.warpAffine`) | Uniform, ±15° | Handheld camera tilt / imperfect alignment |
| Brightness shift | Multiplicative scaling of the HSV **Value** channel | Uniform, 0.6× – 1.4× | Ambient illumination variance (indoor/outdoor/flash) |
| Sensor noise | Additive zero-mean Gaussian noise on the BGR image | σ ∈ [5, 20] | Smartphone image-sensor / low-light noise |

### 4.1 Simulated Illumination Conditions

The brightness-shift factor is binned into three qualitative lighting
regimes for reporting purposes. Because the factor is sampled from a
continuous uniform distribution, the proportion of samples in each
regime follows directly from the width of its interval:

| Illumination Condition | Brightness Factor Range | Approx. Share of Augmented Samples |
|---|---|---:|
| Low-light / underexposed | 0.60× – 0.85× | ~31% |
| Normal / ambient | 0.85× – 1.15× | ~38% |
| Bright / overexposed | 1.15× – 1.40× | ~31% |

This distribution ensures the augmented portion of the set spans a
representative spread of lighting conditions, directly addressing the
illumination-sensitivity concern that motivates the Hue-Saturation-only
(Value-excluded) histogram design of the color inspection module.

## 5. Seed-to-Augmented Image Mapping (Reproducibility)

Each of the 1,600 output images is traceable to exactly one source seed
photograph, preserving a clean provenance chain from raw Kaggle image →
seed photograph → (optionally) augmented derivative:

1. **Originals retained.** Every seed image is copied into the output
   set unmodified first.
2. **Even derivative distribution.** The remaining images needed to
   reach the 800-per-class target are distributed evenly across seed
   images, so no single source photograph dominates the augmented
   portion.
3. **Deterministic random seed.** All random draws (rotation angle,
   brightness factor, noise σ, and the gallery/eval split itself) are
   generated from a fixed master seed (`RANDOM_SEED = 42`), so
   re-running `sort_kaggle_dataset.py` + `augment_data.py` on the same
   raw Kaggle download regenerates an identical dataset byte-for-byte.
4. **Explicit manifest.** Every output directory (`dataset/genuine/`,
   `dataset/counterfeit/`) contains a `_manifest.csv` recording, for
   every generated file: its filename, the source seed image, whether it
   is an `original` or `augmented` sample, and the exact
   rotation/brightness/noise parameters used.

**Naming convention:**
`<class>_<index>_seed<seed_id>_<orig|aug<n>>.jpg`

## 6. Summary Statistics

| Metric | Value |
|---|---|
| Source | Kaggle "Fake Brand Dataset" (prashanthgolla), CC BY-NC 4.0 |
| Brands covered | Nike, Puma, Adidas, Gucci, LV |
| Raw images (5 brands) | 855 genuine, 710 counterfeit (1,565 total) |
| Held-out reference gallery | 68 images (genuine only) |
| Final evaluation set | 800 genuine, 800 counterfeit (1,600 total, 1:1) |
| Share of final set that is real (unaugmented) photography | 93.6% (1,497 / 1,600) |
| Rotation range | ±15° |
| Brightness range | 0.6× – 1.4× |
| Gaussian noise σ range | 5 – 20 |
| Random seed (reproducibility) | 42 |
