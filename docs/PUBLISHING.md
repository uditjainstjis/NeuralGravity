# Publishing the Results Page

This repo includes a static GitHub Pages site at [`index.html`](./index.html).

## Enable GitHub Pages

In the GitHub repository:

1. Open `Settings`
2. Open `Pages`
3. Under `Build and deployment`, choose `Deploy from a branch`
4. Select:
   - Branch: `main`
   - Folder: `/docs`
5. Save

Expected site URL:

```text
https://uditjainstjis.github.io/NeuralGravity/
```

## What the page shows

- current speculative decoding result
- draft-length `K` sweep
- TTA* benchmark summary
- practical lessons from the experiments

## Updating the page

If benchmark artifacts change, update:

- [`index.html`](./index.html)
- [`../reports/RESULTS_SUMMARY.md`](../reports/RESULTS_SUMMARY.md)
- [`../reports/iclr_2026_submission.tex`](../reports/iclr_2026_submission.tex)

The page is static and does not read JSON at runtime, so benchmark numbers should be edited manually to match the saved artifacts.
