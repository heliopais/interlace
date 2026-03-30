# profile_confint

Compute profile likelihood confidence intervals for variance parameters
(the theta / relative Cholesky factor scale). Mirrors
`lme4::confint.merMod(method="profile")`.

```{eval-rst}
.. autofunction:: interlace.profile_ci.profile_confint
```

## Example

```python
import interlace
from interlace.profile_ci import profile_confint

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])

ci = profile_confint(result, level=0.95)
print(ci)
#                   estimate    2.5 %   97.5 %
# subject.(Int)       0.671    0.412    0.934
# item.(Int)          0.346    0.180    0.541
```

## Notes

CIs are reported on the **theta** (relative Cholesky factor) scale.
For intercept-only random effects, `sigma_b ≈ theta * sqrt(sigma2_hat)`.
If the profile drops below the target before theta reaches zero, the lower
bound is set to 0 (boundary case).

Always uses ML (not REML) internally for the profile likelihood, regardless
of how the model was originally fitted.

## See also

- {doc}`convergence` — `isSingular()` for detecting boundary fits
- [Variance Inference Guide](../variance-inference.md) — choosing between bootstrap and profile CIs
