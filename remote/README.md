# remote

Setup for training on a rented GPU box (e.g. [vast.ai](https://vast.ai)) over SSH, as an
alternative to `colab/run_maskplace.ipynb` once you've used up Colab's free-tier limits.

## vast.ai quickstart

1. Rent an instance (a template with CUDA + Python is enough; no Jupyter template needed).
2. `ssh` in (vast.ai gives you the command on the instance page), then:

   ```sh
   git clone https://github.com/paulin-dev/placax.git && cd placax
   ./remote/setup.sh
   ```

3. Run training inside `tmux` so it survives a dropped SSH connection:

   ```sh
   tmux new -s train
   ./remote/train.sh                      # defaults: benchmarks/adaptec1, 300 iterations
   # Ctrl-b d to detach; `tmux attach -t train` later to check back in
   ```

4. Pull the checkpoint/plots back to your machine with `rsync` or `scp` (see the command
   `train.sh` prints at the end), then **destroy the instance** — vast.ai bills per hour and
   the disk isn't kept once you do.

`train.sh` re-runs `run_maskplace.py`'s own checkpoint-resume, so stopping and re-running it
continues training rather than restarting.
