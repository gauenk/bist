# Bayesian-Inspired Space-Time Superpixels (BIST)

This project presents Bayesian-inspired Space-Time Superpixels (BIST): a fast, state-of-the-art method to compute space-time superpixels. BIST is a novel extension of a single-image Bayesian method named BASS, and it is inspired by hill-climbing to a local mode of a Dirichlet-Process Gaussian Mixture Model (DP-GMM). The method is only Bayesian-inspired, rather than actually Bayesian, because it includes heuristic modifications to the theoretically correct sampler. Similar to existing methods, BIST can adapt the number of superpixels to an individual frame using split-merge steps. A key novelty is a new temporal coherence term in the split step, which reduces the chance of splitting propagated superpixels. This term enforces temporal coherence in propagated regions, and unconstrained adaptation in disoccluded regions. A hyperparameter determines the strength of this new term, which does not require special tuning to return consistent results across multiple videos. The wall-clock runtime of BIST is over twice as fast as BASS and over 30 times faster than the next fastest space-time superpixel method with open-source code.

<p align="center">
<img src="assets/kid-football.gif" width="500">
</p>

## Installation

BIST is written in C++/CUDA. The source code can be compiled into an executable binary file and/or a Python API.

1. Clone the repository:
    ```bash
    git clone https://github.com/gauenk/bist.git
    cd bist
    ```

2. Install the executable binary:
    ```bash
    mkdir build
    cmake ..
    make -j8
    ```

3. (and/or) Install the Python API:
    ```bash
    pip install -r requirements.txt
    python -m pip install -e .
    ```

## Usage

### Running the Compiled Binary

To run the compiled binary file, an example command is given below:

```bash
./bin/bist -d video_directory/ -f flow_directory/ -o output_directory/ -n 25 --iperc_coeff 4.0 --thresh_new 0.05 --read_video 1
```

The example highlights important arguments such as io directories, initial superpixel size (n), the temporally coherent split step hyperparameter (iperc_coeff, or $\gamma$), and the threshold to relabel a propogated superpixel as a new one (thresh_new, or $\varepsilon_{\text{new}}$). The last input (read_video) determines if the algorithm uses BIST (==1) or BASS (==0).

We've included two clips to validate the installation. One clip is from the [SegTrack v2 dataset](https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html) and the other clip is from the [DAVIS](https://davischallenge.org/) dataset. Example commands to run BIST on these clips is given below:

```bash
./bin/bist -d data/examples/hummingbird/imgs/ -f data/examples/hummingbird/flows/ -o results/hummingbird/ -n 25 --iperc_coeff 4.0 --thresh_new 0.05 --read_video 1
```

```bash
./bin/bist -d data/examples/kid-football/imgs/ -f data/examples/kid-football/flows/ -o results/kid-football/ -n 25 --iperc_coeff 4.0 --thresh_new 0.05 --read_video 1
```

### Using the Python API

An example script to use the BIST Python API is detailed below:

```python
import bist
import torch as th

# Read the video & optical flow
vid = bist.utils.read_video("examples/kid-football/imgs")
flows = bist.utils.read_flows("examples/kid-football/flows")
# vid.shape = (T,H,W,C)
# flows.shape = (T,H,W,2)

# Run BIST
kwargs = {"n":25,"read_video":True,"iperc_coeff":4.0,"thresh_new":thresh_new,'rgb2lab':True}
spix = bist.run(vid,flows,**kwargs)

# Mark the image with the superpixels
color = th.tensor([0.0,0.0,1.0]) # color of border
marked = bist.get_marked_video(vid,spix,color)

# Computer the superpixel-pooled video
pooled = bist.get_pooled_video(video,spix)

# Save or display the denoised image
bist.utils.save_spix(spix, 'results/kid-football',"%05d.csv")
bist.utils.save_video(marked, 'results/kid-football',"border_%05d.png")
bist.utils.save_video(pooled, 'results/kkid-football',"pooled_%05d.png")
```

_Note_: BIST uses *forward* optical flows. [The flow files for some sequences can be downloaded here](https://drive.google.com/drive/folders/1598mrD5gSSM-cYLeNOnLTeXDwE2zYCVU?usp=sharing), but they are too big to host them all.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{gauen2025bist,
  title={Bayesian-Inspired Space-Time Superpixels},
  author={Gauen, Kent W and Chan, Stanley H},
  journal={prepint},
  year={2025}
}
```
