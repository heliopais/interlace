# Leverage

Compute the hat-matrix diagonal decomposed into fixed-effect and random-effect
components, following Demidenko & Stukel (2005) and Nobre & Singer (2007).

```{eval-rst}
.. autofunction:: interlace.leverage
```

## Returned columns

| Column | Description |
|---|---|
| `overall` | H1 + H2 (total leverage) |
| `fixef` | H1 — fixed-effect leverage |
| `ranef` | H2 — random-effect leverage (Demidenko & Stukel) |
| `ranef.uc` | Unconfounded H2 (Nobre & Singer) |

## Example

```python
import interlace

result = interlace.fit("rt ~ condition", data=df, groups=["subject", "item"])
lev = interlace.leverage(result)

# Flag high-leverage observations (rule of thumb: overall > 2p/n)
p = len(result.fe_params)
n = result.nobs
lev_high = lev[lev["overall"] > 2 * p / n]
print(f"{len(lev_high)} high-leverage observations")
```

## See also

- {doc}`influence` — Cook's distance and MDFFITS (impact on estimates, not just fit)
- {doc}`augment` — append leverage + influence metrics to the original DataFrame
- {doc}`residuals` — residual diagnostics
