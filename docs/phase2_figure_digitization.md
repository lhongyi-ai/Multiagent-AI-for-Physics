# Phase 2 Figure Digitization

Values that appear only in plots are not estimated automatically.

The acquisition pipeline creates digitization tasks for figure-only evidence, especially optical spectral-weight quantities such as:

- `S_delta / S_n`
- `S_u / S_n`
- missing optical spectral weight
- optical sum-rule redistribution

Digitized CSV imports must include:

```text
doping_x,observable_value,x_uncertainty,y_uncertainty,series_label,digitization_method,reviewer,source_figure
```

Figure-derived rows remain review-gated and are not auto-promoted by default.
