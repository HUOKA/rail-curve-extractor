# Data Layout

This repository does not track production DOM, DSM, LAS, CVAT exports, model checkpoints, or generated outputs.

Place local data under `data/` or pass explicit paths to the scripts. The full Tonghaigang workflow expects DJI Terra-style products similar to:

```text
data/production/
  terra_dom/
    dom.tif
    dom.tfw
    dom.prj
  terra_dsm/
    dsm.tif
    dsm.tfw
    dsm.prj
  terra_las/
    cloud0.las
    cloud1.las
```

The real data is intentionally ignored by Git. Keep only small configuration examples in the repository.
