# Bayesian-Inspired Space-Time Superpixels (BIST)

Bayesian-Inspired Space-Time Superpixels (BIST) is a fast, temporally-coherent space-time superpixel method that achieves state-of-the-art benchmark results. The method is Bayesian-Inspired, rather than actually Bayesian, since the split step uses a heuristic modification of the original Hastings ratio. This modification is a core novelty of our method, and dramtically reduces the number of superpixels. BIST is frequently more than twice as fast as BASS, and is over 30 times faster than other space-time superpixel methods with favorable (and sometimes superior) quality. Specifically, BIST runs at 60 frames per second while TSP runs at about 2 frames per second. Additionally, to garner interest in superpixels, this paper demonstrates their use within deep neural networks. We present a superpixel-weighted convolution layer for single-image denoising that outperforms standard convolution by over 1.5 dB PSNR.

<p align="center">
<img src="assets/kid-football.gif" width="500">
</p>

## Installation

BIST is written in C++/CUDA. The source code can be compiled into an executable binary file or a Python API (or both).

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

3. Install the Python API:
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

_Note_: BIST uses *forward* optical flows.

## Superpixel Convolution

To demonstrate the value of superpixels within deep neural networks, we present a novel layer called superpixel convolution. While more expensive than traditional convolution, this module significantly outperforms standard convolution on the example application of image denoising (by over 1.5 dB PSNR). Details about this portion of the code base will be added soon (noted April 2025).

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
