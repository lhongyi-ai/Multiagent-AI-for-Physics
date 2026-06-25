# Phase 2 Data Schema

The canonical compatibility file remains `data/phase2_lsco.csv` with one observable per row:

```text
observation_id,material_family,material_id,doping,observable,value,unit,uncertainty,split,source_id,provenance,usable_for_fit,source_url,curation_note
```

Unreviewed acquisition output is normalized JSONL, not directly appended to the canonical CSV. Candidate rows include:

- candidate row ID
- material identity
- doping string and parsed `doping_x`
- observable
- normalized value and unit
- raw value and unit
- uncertainty
- paper ID, DOI, arXiv ID, and source URL
- page/table/figure/evidence text
- extraction method
- confidence
- validation status and review status

This avoids mixing measurements from different samples or papers while preserving backward compatibility with existing Phase 2 tools.
