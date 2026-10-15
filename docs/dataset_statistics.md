# Dataset Acquisition & Characteristics

## 0. Exact Dataset Statistics (Summary)

All figures below are computed directly from `dataset/genuine/_manifest.csv`
and `dataset/counterfeit/_manifest.csv` — the per-image provenance ledger
written by `augment_data.py` for every run — not estimated or rounded.
Regenerate the manifests with `python3 sort_kaggle_dataset.py ...` followed
by `python3 augment_data.py --target 800` (fixed seed `42`) to reproduce
these exact counts.

| # | Statistic | Value |
|---|---|---|
| 1 | Number of genuine samples | **800** |
| 2 | Number of counterfeit samples | **800** |
| 3 | Images per category | Genuine: 800 (787 unmodified + 13 augmented). Counterfeit: 800 (710 unmodified + 90 augmented) |
| 4 | Samples per illumination condition | See Table 0.1 below — computed from the *realized* brightness factor of every augmented image, not the theoretical sampling distribution |
| 5 | Genuine : Counterfeit ratio | **1 : 1** (800 : 800, exact) |
| 6 | Do multiple images originate from the same physical product? | **Yes, within a seed group** (an original photo and its augmented derivatives share one physical unit) — see §0.2 for the precise, honest answer, including what is *not* verifiable |

### 0.1 Illumination Conditions — Exact Realized Counts

Only the 103 **augmented** images (6.4% of the 1,600-image set) carry a
simulated illumination perturbation; the 1,497 unmodified originals retain
whatever real-world lighting was present in the source photograph and are
not assigned an illumination label. Counts below are the *actual* bucket
membership of every augmented image's realized brightness factor, not the
theoretical ~31/38/31% split implied by the sampling range widths:

| Illumination Condition | Brightness Factor Range | Genuine (n=13) | Counterfeit (n=90) | Total (n=103) | Share of Augmented |
|---|---|---:|---:|---:|---:|
| Low-light / underexposed | 0.60× – 0.85× | 4 | 19 | 23 | 22.3% |
| Normal / ambient | 0.85× – 1.15× | 7 | 44 | 51 | 49.5% |
| Bright / overexposed | 1.15× – 1.40× | 2 | 27 | 29 | 28.2% |
| **Total** | — | **13** | **90** | **103** | **100%** |

At n=103 draws, realized proportions deviate from the theoretical
interval-width split (31/38/31%) by ordinary sampling variance — this is
expected and is why we report the *realized*, auditable counts here rather
than the theoretical distribution alone.

### 0.2 Physical-Product Provenance — What Is and Isn't Verifiable

This question has two distinct parts, and conflating them is exactly the
kind of reproducibility gap that needs to be stated explicitly rather than
implied:

**Within this pipeline (fully verifiable):** Every augmented image is a
geometric/photometric transform of exactly one original seed photograph,
recorded in the `source_seed_image` column of the manifest. In that sense,
yes — an original image and its augmented derivative(s) are multiple
images of the *same physical capture* of the same physical unit. Across
the 1,600-image set: 787 genuine images are traceable to 787 unique
genuine seed photographs (13 of those seeds contributed a second,
augmented image each — max 2 outputs per seed); 800 counterfeit images
are traceable to 710 unique counterfeit seed photographs (90 seeds
contributed a second image each). No seed photograph contributed more
than 2 output images.

**Across different raw seed photographs (not verifiable from the source
data):** Whether two *different* raw photographs from the Kaggle "Fake
Brand Dataset" — e.g. `nike_0001.jpg` and `nike_0002.jpg` — depict the
same physical pair of shoes from a different angle, or two entirely
different physical pairs, is **not documented by the dataset's
publisher** and cannot be independently verified from filenames or
metadata alone. This is stated as an explicit limitation (see the
"Threats to Validity" section of the accompanying paper, subsection
"Label provenance") rather than assumed away. Each raw photograph is
therefore treated as an independent sample for evaluation purposes; if a
subset of raw photographs does in fact depict the same physical units,
the effective diversity of the dataset would be somewhat lower than the
raw image count suggests. This does not affect the genuine/counterfeit
label itself (which the publisher does assert per-image), only the
strength of the independence assumption between samples.

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

| Class | Seed Images (unmodified) | Augmented Images | Total Images | Class Ratio |
|---|---:|---:|---:|---|
| Genuine (Authentic) | 787 | 13 | 800 | 1 : 1 |
| Counterfeit | 710 | 90 | 800 | 1 : 1 |
| **Total** | **1,497** | **103** | **1,600** | — |

93.6% of the final evaluation set (1,497 of 1,600 images) is unmodified
real photography; augmentation contributes only the minimal top-up
needed for an exact 1:1, round-number balance. This is a deliberate
inversion of the small-seed-corpus regime (where augmentation must do
most of the work) once a sufficiently large real corpus is available.

**Which seeds receive the top-up is itself randomized and seeded.**
`augment_data.py` selects which specific seed photographs receive an
extra augmented derivative via `random.Random(42).sample(...)` over the
full seed list — not simply the first *N* seeds in file order. This
matters in practice: brand subfolders are sorted alphabetically
(`adidas_*` before `gucci_*`, `lv_*`, `nike_*`, `puma_*`), so a
naive "first N seeds" rule would silently concentrate every top-up image
into whichever brand happens to sort first, distorting the per-brand
composition without any error or warning. The random-sample selection
avoids this: in the current build, the 103 top-up images are spread
across multiple brands rather than concentrated in one (see the
per-brand table in the accompanying paper, Table VIII, for the resulting
per-brand evaluation counts).

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

### 4.1 Simulated Illumination Conditions — Realized Counts

See §0.1 above for the exact, manifest-derived counts per condition
(23 low-light, 51 normal, 29 bright, out of 103 augmented images). The
brightness-shift factor is sampled from a continuous uniform
distribution over [0.6×, 1.4×]; at n=103 draws the realized split departs
somewhat from the theoretical interval-width proportions (~31/38/31%),
which is expected sampling variance and is why the manifest-derived
counts, not the theoretical distribution, are the authoritative figure.

This spread ensures the augmented portion of the set spans a
representative range of lighting conditions, directly addressing the
illumination-sensitivity concern that motivates the Hue-Saturation-only
(Value-excluded) histogram design of the color inspection module.

## 5. Seed-to-Augmented Image Mapping (Reproducibility)

Each of the 1,600 output images is traceable to exactly one source seed
photograph, preserving a clean provenance chain from raw Kaggle image →
seed photograph → (optionally) augmented derivative:

1. **Originals retained.** Every seed image is copied into the output
   set unmodified first.
2. **Randomized, seeded top-up selection.** The images needed to reach
   the 800-per-class target are assigned to a random sample of seed
   images (seeded, not "first N in file order" — see Section 3), so no
   single brand or source photograph is silently over-represented.
3. **Deterministic random seed.** All random draws (which seeds get a
   top-up, rotation angle, brightness factor, noise σ, and the
   gallery/eval split itself) are generated from a fixed master seed
   (`RANDOM_SEED = 42`), so re-running `sort_kaggle_dataset.py` +
   `augment_data.py` on the same raw Kaggle download regenerates an
   identical dataset byte-for-byte.
4. **Explicit manifest.** Every output directory (`dataset/genuine/`,
   `dataset/counterfeit/`) contains a `_manifest.csv` recording, for
   every generated file: its filename, the source seed image, whether it
   is an `original` or `augmented` sample, and the exact
   rotation/brightness/noise parameters used. This is the file §0 above
   was computed from directly.

**Naming convention:**
`<class>_<index>_seed<seed_id>_<orig|aug<n>>.jpg`

## 6. Summary Statistics

| Metric | Value |
|---|---|
| Source | Kaggle "Fake Brand Dataset" (prashanthgolla), CC BY-NC 4.0 |
| Brands covered | Nike, Puma, Adidas, Gucci, LV |
| Raw images (5 brands) | 855 genuine, 710 counterfeit (1,565 total) |
| Held-out reference gallery | 68 images (genuine only) |
| Final evaluation set | 800 genuine, 800 counterfeit (1,600 total, 1:1 exact) |
| Share of final set that is real (unaugmented) photography | 93.6% (1,497 / 1,600) |
| Illumination-labeled subset | 103 images (23 low-light, 51 normal, 29 bright) |
| Rotation range | ±15° |
| Brightness range | 0.6× – 1.4× |
| Gaussian noise σ range | 5 – 20 |
| Random seed (reproducibility) | 42 |
| Max output images per single seed photograph | 2 |
