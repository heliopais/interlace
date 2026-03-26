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
