# MATLAB baseline: CDNN

## What this is

This directory holds the correlation-driven neural network (CDNN) baseline from Samavatian et al. (2020). It is the reference model that the Python models in this repo are compared against.

The baseline is not ported to Python. It stays in MATLAB, unmodified from the authors' original supplement.

## Why MATLAB, why not ported

The handover decision for this repo: the CDNN baseline stays in MATLAB. Porting it to Python would introduce a second source of implementation error and make the comparison less honest. The point of the baseline is that it is the same code the original authors published, run on the same data.

If the CDNN were reimplemented in Python, any difference in results could be attributed to the port rather than to the model. Keeping the baseline in MATLAB removes that variable.

## Source and attribution

The code is the supplementary source released with:

> V. Samavatian, M. Fotuhi-Firuzabad, M. Samavatian, P. Dehghanian, and F. Blaabjerg, "Correlation-driven machine learning for accelerated reliability assessment of solder joints in electronics," *Scientific Reports*, vol. 10, no. 1, p. 14821, 2020.

Author of the original code: Vahid Samavatian, Sharif University of Technology. The header of `CDNN_Main_File.m` carries the original authorship notice. It is preserved as-is.

Licensing: CC BY 4.0, same as the source paper. See `data/README.md` for the full licence statement. Attribution to the authors is required when this code is redistributed.

## Files

| File | Role |
|------|------|
| `CDNN_Main_File.m` | Entry point. Loads data, scales it, trains the network, saves weights, evaluates. |
| `trainConv.m` | Training loop. Adjusts the correlation matrices and the fully connected layer weights. |
| `Conv.m` | Correlation layer. Applies the trained correlation matrices to the input. |
| `Pool.m` | Average pooling layer. |
| `ReLU.m` | ReLU activation. |
| `Sigmoid.m` | Sigmoid activation. |
| `Scale.m` | Min-max scaling to a given range. |
| `predict.m` | Forward pass and evaluation. Returns RMSE, per-sample error, and Pearson correlation. |
| `rng.m` | Custom random number helper. Shadows MATLAB's built-in `rng` when this directory is on the path. |

The `.mat` files (`CDNN.mat`, `weights.mat`) are outputs of a training run. They are not committed. They are listed in `.gitignore`.

## Requirements

- MATLAB. The entry script uses `normalize`, which was introduced in R2018a, so R2018a or later is required.
- No MATLAB toolboxes are required beyond base MATLAB. All helper functions are in this directory.

## Data layout

The entry point expects a folder named `Data` next to the scripts, containing `.txt` files. Each file is read as a `7 x 4` matrix:

- Rows 1 to 6, all 4 columns: 24 feature values for one sample.
- Row 7, column 1: measured lifetime in hours.

One file equals one sample. For the real dataset, that is 450 files in `Data/`.

## How to run

1. Copy or symlink the raw data into `baseline_matlab/Data/`.

   On macOS or Linux, from inside `baseline_matlab/`:
   ```
   ln -s ../data/raw Data
   ```

   On Windows, from inside `baseline_matlab/` in a command prompt with developer mode enabled:
   ```
   mklink /J Data ..\data\raw
   ```

   Or simply copy the files:
   ```
   mkdir Data
   cp ../data/raw/*.txt Data/
   ```

   `baseline_matlab/Data/` is gitignored. It is a local staging step.

2. Open MATLAB, navigate to `baseline_matlab/`, and run:
   ```
   CDNN_Main_File
   ```

3. For the synthetic dataset, create a second staging folder named `Synth_Data/` the same way, pointing at `data/synthetic/`. Then change line 27 of `CDNN_Main_File.m` from `'Data'` to `'Synth_Data'`. That line is intended to be edited locally to switch between datasets. Do not commit the edit.

4. The script prints `RMSE`, a per-sample `Error` vector, and the Pearson correlation `r`.

## Expected output

The script prints three values to the MATLAB console:

- `RMSE`: root mean squared error between predicted and measured lifetime, on the scaled range.
- `Error`: vector of per-sample errors.
- `r`: Pearson correlation between predicted and measured lifetime.

The thesis reports R² of approximately 0.821 for the CDNN on real data and 0.613 on synthetic data. Those numbers are the reference points the Python models are benchmarked against. Exact reproduction depends on the MATLAB version, the random initialisation, and the machine.

## Generated artefacts

Running `CDNN_Main_File.m` writes two files into `baseline_matlab/`:

- `CDNN.mat`: input data, scaled data, all weights, and denormalisation constants.
- `weights.mat`: weights only.

Both are training outputs. Both are gitignored. They do not need to be committed.

## Known caveats

1. The script does not fix the random seed. Results may vary slightly between runs. The thesis reports one run.
2. Training is single-threaded on CPU unless MATLAB is configured otherwise. Runtime depends on hardware.
3. The synthetic data switch requires editing one line of the entry script. This is documented above. It is a deliberate local edit, not a repo change.
4. The `.mat` outputs are not part of the repo. Exact reproduction of the published CDNN numbers requires rerunning the script on the same hardware and MATLAB version, or obtaining the artefacts from the thesis author.
5. The correlation architecture (`CC = [3 3 80]`) and hidden layers (`[640 580 500]`) are fixed in the entry script. Changing them deviates from the published baseline and breaks the comparison.

## License

The code in this directory is distributed under CC BY 4.0, inherited from the Samavatian et al. paper's supplementary material. The header of `CDNN_Main_File.m` credits the original authors.

Code written by the thesis author in `src/` is licensed under Apache 2.0. See `LICENSE` at the repo root.