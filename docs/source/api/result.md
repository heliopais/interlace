# CrossedLMEResult

The result object returned by {func}`interlace.fit`. Its attribute names and structure
mirror `statsmodels.MixedLMResults`, making it a drop-in replacement for downstream
code.

```{eval-rst}
.. autoclass:: interlace.CrossedLMEResult
   :members:
   :undoc-members:
```

## ModelInfo

Internal container for model matrices and metadata, accessible as `result.model`.

```{eval-rst}
.. autoclass:: interlace.result.ModelInfo
   :members:
   :undoc-members:
```
