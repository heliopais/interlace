# quantreg_ker_se / ols_dfbetas_qr

Low-level quantile regression utilities used internally by the influence
diagnostics pipeline. Exposed publicly for users who need to compute
kernel-based standard errors or QR-based DFBETAS on their own quantile
regression fits.

## quantreg_ker_se

```{eval-rst}
.. autofunction:: interlace.quantreg_ker_se
```

## ols_dfbetas_qr

```{eval-rst}
.. autofunction:: interlace.ols_dfbetas_qr
```

## Notes

These functions replicate the behaviour of R's `quantreg::summary.rq(se="ker")`
and the DFBETAS diagnostic from `car::dfbetas()`. They are used internally by
`hlm_influence()` and `lmer_influence_measures()`.

## See also

- {doc}`influence` — high-level influence diagnostics
- {doc}`augment` — combined augmented DataFrame with `.cooksd` and `.mdffits`
