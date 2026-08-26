
<h1 align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.svg">
        <source media="(prefers-color-scheme: light)" srcset="assets/logo_light.svg">
        <img alt="placax" src="assets/logo_light.svg" width="300">
    </picture><br>
    <b>A fast, shared JAX environment for chip macro placement research</b>
</h1>

<p align="center">
  <a href="https://pypi.python.org/pypi/placax"><img src="https://img.shields.io/pypi/pyversions/placax.svg?style=flat" /></a>
  <a href= "https://badge.fury.io/py/placax"><img src="https://badge.fury.io/py/placax.svg" /></a>
  <a href= "https://github.com/paulin-dev/placax/blob/master/LICENSE.md"><img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
</p>

This is the detailled description



GitHub desc: A shared, differentiable JAX environment for chip macro placement


## Architecture

<img src="assets/placax_architecture.svg" alt="placax architecture" width="100%">

## Getting Started

```sh
pip install placax
```

## Training (MaskPlace pipeline)

```sh
python scripts/run_maskplace.py --benchmark_dir=benchmarks/adaptec1 --n_iterations=300 --n_episodes=auto
```

- `--benchmark_dir`: path to a downloaded benchmark (see `scripts/download_benchmarks.py`); default `benchmarks/adaptec1`.
- `--n_iterations`: number of buffered-PPO update cycles to run; default `100`.
- `--macro_budget`: place only the N most important macros (MaskPlace's `--pnm`); default `128`, MaskPlace's own value. Pass `all` to place every macro in the netlist instead - not yet verified to fit in memory or train well at that scale.
- `--n_episodes`: episodes collected per PPO update (MaskPlace's own default is `10`); pass `auto` to auto-detect the largest that fits on your GPU instead of picking a number yourself.

Run `python scripts/run_maskplace.py --help` for the full flag list. The script auto-resumes from its checkpoint on re-run, so it's safe to stop and restart with the same flags.