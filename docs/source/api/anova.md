# anova

Likelihood-ratio test for comparing two nested linear mixed models.
Mirrors `lme4::anova.merMod()` output format.

```{eval-rst}
.. autofunction:: interlace.anova
```

## Example

```python
import interlace

# Both models must use method="ML"
m0 = interlace.fit("rt ~ 1",         data=df, groups=["subject", "item"], method="ML")
m1 = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"], method="ML")

result = interlace.anova(m0, m1)
print(result)
#        Df       AIC       BIC       logLik  deviance  Chisq  Chi Df  Pr(>Chisq)
# Model 1  3  1234.56  1245.67  -614.28   1228.56
# Model 2  4  1228.12  1242.45  -610.06   1220.12  8.44       1    0.0037
```

## Notes

Both models must have been fitted with `method="ML"`. When comparing models
that differ only in their random-effect structure, REML is acceptable and
often preferred; use `method="REML"` in that case and pass both models with
the same fixed effects.

## See also

- {doc}`fit` — `method="ML"` parameter
- [Model Comparison Guide](../model-comparison.md) — full LRT workflow
