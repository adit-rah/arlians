# data/ — generated artifacts (not source)

Output directory for generated worlds and rendered media. **Everything here is reproducible
output — don't treat it as source, and don't hand-edit it.**

- `world_<seed>/` — exported worlds from `generate.py` (consumed by the `stream/` server and
  as a cache). Regenerate with `python generate.py --seed <n> --output ./data/world_<n>`.
- `*.gif` / `*.png` — rollout/demo renders and phase sanity frames from `scripts/demo.py` and
  the render tests.

Default output paths point here (`scripts/demo.py --out data/demo.gif`,
`generate.py --output ./data/world_<seed>`). Safe to delete anything here and regenerate.
Large binaries — consider whether they belong in git before committing new ones.
