# Dataset Acquisition & Characteristics

## 1. Data Acquisition

The evaluation dataset used in this study was constructed from a small
*seed* corpus of physically photographed product samples: **10 genuine
(authentic) product images** and **10 counterfeit product images**,
captured under uncontrolled, real-world smartphone/webcam conditions
(handheld capture, ambient indoor lighting, MacBook Air FaceTime camera
and/or smartphone camera). Each seed image corresponds to a distinct
physical unit or a distinct viewing angle of a physical unit, and serves
as the ground-truth reference for one class.

Because 10 samples per class are insufficient to obtain a statistically
meaningful estimate of system accuracy, the seed corpus was expanded
using a controlled, reproducible **data augmentation pipeline**
(`augment_data.py`) to a final evaluation set of **500 genuine and 500
counterfeit images (1,000 images total)**. This mirrors standard
practice in small-sample computer-vision studies, where geometric and
photometric augmentation is used to approximate the natural variance of
repeated real-world image capture (viewing angle, ambient light, sensor
noise) without requiring the collection of hundreds of additional
physical samples.

## 2. Dataset Composition

| Class | Seed Images | Augmented Images | Total Images | Class Ratio |
|---|---:|---:|---:|---|
| Genuine (Authentic) | 10 | 490 | 500 | 1 : 1 |
| Counterfeit | 10 | 490 | 500 | 1 : 1 |
| **Total** | **20** | **980** | **1,000** | — |

The dataset is class-balanced by construction (1:1 genuine-to-counterfeit
ratio), which avoids the accuracy-inflation bias that an imbalanced
dataset would introduce and allows Accuracy, Precision, and Recall to be
interpreted directly without re-weighting.

## 3. Augmentation Pipeline & Simulated Capture Conditions

Each seed image is expanded through three independent, randomly
parameterized perturbations applied in sequence, chosen to approximate
the dominant sources of variance in repeated smartphone photography of a
physical object:

| Perturbation | Method | Parameter Range | Simulates |
|---|---|---|---|
| Geometric rotation | Affine rotation about image center (`cv2.warpAffine`) | Uniform, ±15° | Handheld camera tilt / imperfect alignment |
| Brightness shift | Multiplicative scaling of the HSV **Value** channel | Uniform, 0.6× – 1.4× | Ambient illumination variance (indoor/outdoor/flash) |
| Sensor noise | Additive zero-mean Gaussian noise on the BGR image | σ ∈ [5, 20] | Smartphone image-sensor / low-light noise |

### 3.1 Simulated Illumination Conditions

The brightness-shift factor is binned into three qualitative lighting
regimes for reporting purposes. Because the factor is sampled from a
continuous uniform distribution, the proportion of samples in each
regime follows directly from the width of its interval:

| Illumination Condition | Brightness Factor Range | Approx. Share of Augmented Samples |
|---|---|---:|
| Low-light / underexposed | 0.60× – 0.85× | ~31% |
| Normal / ambient | 0.85× – 1.15× | ~38% |
| Bright / overexposed | 1.15× – 1.40× | ~31% |

This distribution ensures the model under test is evaluated across a
representative spread of lighting conditions rather than only the
lighting present at the time the seed photo was captured, directly
addressing the illumination-sensitivity concern that motivates the
Hue-Saturation-only (Value-excluded) histogram design of the color
inspection module.

## 4. Seed-to-Augmented Image Mapping (Reproducibility)

Each of the 1,000 output images is traceable to exactly one physical
seed sample, preserving a clean provenance chain from physical product →
seed photograph → augmented derivative. The mapping is generated as
follows:

1. **Originals retained.** All 10 seed images per class are copied
   into the output set unmodified, contributing 10 zero-perturbation
   samples per class.
2. **Even derivative distribution.** The remaining `500 − 10 = 490`
   images per class are distributed evenly across the 10 seed images
   (`490 ÷ 10 = 49` derivatives per seed), so no single physical sample
   dominates the dataset.
3. **Deterministic random seed.** All random draws (rotation angle,
   brightness factor, noise σ) are generated from a fixed master seed
   (`RANDOM_SEED = 42`), so re-running `augment_data.py` on the same
   seed corpus regenerates an identical dataset byte-for-byte —
   satisfying reproducibility requirements for peer review.
4. **Explicit manifest.** Every output directory (`dataset/genuine/`,
   `dataset/counterfeit/`) contains a `_manifest.csv` recording, for
   every generated file: its filename, the source seed image it was
   derived from, whether it is an `original` or `augmented` sample, and
   the exact rotation/brightness/noise parameters used — providing a
   full, auditable trail from any dataset image back to its physical
   source.

**Naming convention:**
`<class>_<index>_seed<seed_id>_<orig|aug<n>>.jpg`
e.g. `genuine_0037_seed03_aug012.jpg` denotes the 12th augmented
derivative of genuine seed image #3.

## 5. Summary Statistics

| Metric | Value |
|---|---|
| Total samples | 1,000 |
| Genuine : Counterfeit ratio | 1 : 1 (500 : 500) |
| Seed samples per class | 10 |
| Augmentation multiplier | 50× per class |
| Rotation range | ±15° |
| Brightness range | 0.6× – 1.4× |
| Gaussian noise σ range | 5 – 20 |
| Random seed (reproducibility) | 42 |
| Image format | JPEG, 500×500 px (source-dependent) |
