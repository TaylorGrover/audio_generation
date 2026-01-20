# Additive Synthesizer


## Configure
```bash
$ make venv
$ source venv/bin/activate
$ make install 
```

## Test
After installing all requirements in the python virtual environment, run 
```bash
$ pytest
```

```python
import numpy as np
def f(x: int) -> int:
    return x * 2
```

## Sine Series of Random Points Over Equally-Spaced Intervals
$$
s=[s_1,...,s_n]
$$
### Definitions
$$
a_j=\frac{(j-1)\lambda}{n},\quad b_j=\frac{j\lambda}{n},\quad m=\frac{n}{\lambda}\left(s_{j+1}-s_j\right)
$$
$$
\alpha_i=\frac{2\pi i}{\lambda}
$$

$$
\alpha_ka_j=\frac{2\pi k(j-1)}{n},\quad \alpha_k b_j=\frac{2\pi kj}{n}
$$

### Sine Series Coefficients
$$
A_i=\frac{2}{\lambda}\int_0^\lambda s(t)\sin\left(\frac{2\pi it}{\lambda}\right)\mathops{dt}
$$

$$
s_j(t)=\frac{n}{\lambda}\left(s_{j+1}-s_j\right)(t-a_j)+s_j
$$

$$
A_k=\frac{2}{\lambda}\sum_{j=1}^n\int_{a_j}^{b_j}\left(m_j(t-a_j)+s_j\right)\sin(\alpha_k t)\mathops{dt}=\frac{2}{\lambda}\sum_{j=1}^n\left[-\frac{1}{\alpha_k}\left(m_j(t-a_j)+s_j\right)+\frac{m_j}{\alpha_k^2}\sin(\alpha_kt)\right]_{a_j}^{b_j}
$$

$$
=\frac{2}{\lambda}\sum_{j=1}^n\left[-\frac{1}{\alpha_k}\left(\frac{m_j\lambda}{n}+s_j\right)\cos(\alpha_kb_j)+\frac{s_j}{\alpha_k}\cos(\alpha_ka_j)+\frac{m_j}{\alpha_k^2}\sin(\alpha_kb_j)-\frac{m_j}{\alpha_k^2}\sin(\alpha_ka_j)\right]
$$
