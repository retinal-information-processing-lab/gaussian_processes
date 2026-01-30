# Data Format Reference

**Authoritative source for**: PNAS dataset format and preprocessing steps

---

## 1. Dataset: PNAS Paper Sorted Data

**File**: `notebooks/PNAS_paper_sorted_data.npz`

| Key | Shape | Dtype | Range | Description |
|-----|-------|-------|-------|-------------|
| `images_train` | (2910, 108, 108, 1) | float32 | [-2.4, 2.5] | Training images (normalized) |
| `images_val` | (250, 108, 108, 1) | float32 | [-2.4, 2.5] | Validation images |
| `images_test` | (30, 108, 108, 1) | float32 | [-2.4, 2.5] | Test images |
| `responses_train` | (2910, 41) | float64 | [0, ~25] | Spike counts (41 neurons) |
| `responses_val` | (250, 41) | float64 | [0, ~26] | Validation responses |
| `responses_test` | **(30, 30, 41)** | float64 | [0, ~27] | **30 repeats x 30 images x 41 neurons** |

**Important notes**:
- Images are already normalized (mean ~0, std ~1)
- Responses are non-negative spike counts
- Test set has 30 repetitions per image for reliability estimates
- We fit **one neuron at a time** (cellid selects which)

---

## 2. Preprocessing Steps

From `scripts/one_cell_fit.py`:

```python
# 1. Load data
data = np.load('notebooks/PNAS_paper_sorted_data.npz')
X_train = torch.tensor(data['images_train'], dtype=torch.float32, device=device)
R_train = torch.tensor(data['responses_train'], dtype=torch.float32, device=device)

# 2. Flatten images: (n_samples, 108, 108, 1) -> (n_samples, 11664)
X_train = X_train.reshape(X_train.shape[0], -1)  # (2910, 11664)

# 3. Select single neuron
cellid = 0
r = R_train[:, cellid]  # (2910,)

# 4. Select inducing points (random subset)
ntilde = 100  # or up to 2100
indices = torch.randperm(X_train.shape[0])[:ntilde]
xtilde = X_train[indices]  # (ntilde, 11664)

# 5. Initialize hyperparameters
theta = {
    'sigma_0': torch.tensor(1.0),
    'Amp': torch.tensor(1.0),
    'eps_0x': torch.tensor(0.0),
    'eps_0y': torch.tensor(0.0),
    '-2log2beta': torch.tensor(-2 * np.log(2 * 0.1)),  # beta=0.1
    '-log2rho2': torch.tensor(-np.log(2 * 0.1**2)),    # rho=0.1
}

# 6. Initialize firing rate parameters
f_params = {
    'logA': torch.log(torch.tensor(0.01)),  # A = 0.01
    'lambda0': torch.tensor(1.0),
}
```

---

## 3. Pixel Coordinate Grid

For an image with `n_px_side=108` pixels per side:

```python
# Normalized grid on [-1, 1] x [-1, 1]
ycord, xcord = torch.meshgrid(
    torch.linspace(-1, 1, n_px_side),  # n_px_side = 108 for PNAS
    torch.linspace(-1, 1, n_px_side),
    indexing='ij'
)
xcord = xcord.flatten()  # (n_px_side^2,) = (11664,)
ycord = ycord.flatten()
```

- Center of image: `(eps_0x, eps_0y) = (0, 0)`
- Corners: `(+-1, +-1)`

---

*Extracted from CLAUDE.md Sections 9-10, January 2025*
