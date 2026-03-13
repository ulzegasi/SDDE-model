# SDDE-model

Python-wrapped SDDE solar dynamo model, originally based on Julia's
Stochastic Delay Differential Equation tooling.

## Install (editable)

```bash
pip install -e /Users/ulzg/SABC/SDDE-model
```

## Usage

```python
from sdde_model import init_julia, sn, sn_batch, summary_statistics

init_julia()
```

Call `init_julia()` before importing `tensorflow` or other native-library-heavy modules.
That early bootstrap reduces Julia/TensorFlow library conflicts and also forces
`juliacall` to use the pinned Julia project shipped with `sdde_model`.

If you skip the explicit call, `sdde_model` will still initialize Julia lazily on
first use, but the TensorFlow-safe import ordering is best when you call
`init_julia()` yourself near the top of the script.
