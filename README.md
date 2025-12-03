# SurgeNetSeg: Clip-Level Anatomy Segmentation for Minimally Invasive Surgery

![Figure](figures/example_figure.png)

## Installation

Tested on Ubuntu only.

**Prerequisites:**

-   Python 3.8+
-   PyTorch 1.12+ and corresponding torchvision (preferably with CUDA
    GPU support for quick label propagation)

**Clone our repository:**

``` bash
git clone https://github.com/rlpddejong/SurgeNetSeg.git
```

**Install with pip:**

``` bash
cd SurgeNetSeg
pip install -e .
```
------------------------------------------------------------------------

## Pretrained Models

Pretrained model weights for the interactive segmentation tools are
available for download.

**Direct download links:**

-   **cutie-surgesam-50k.pth**\
    https://huggingface.co/TimJaspersTue/SurgeNetSeg/resolve/main/cutie-surgenetseg-50k.pth?download=true
-   **ritm-surgesam-50k.pth**\
    https://huggingface.co/TimJaspersTue/SurgeNetSeg/resolve/main/ritm-surgenetseg-50k.pth?download=true

**Download via command line:**

``` bash
mkdir -p weights
cd weights

wget -O cutie-surgesam-50k.pth "https://huggingface.co/TimJaspersTue/SurgeNetSeg/resolve/main/cutie-surgenetseg-50k.pth?download=true"
wget -O ritm-surgesam-50k.pth  "https://huggingface.co/TimJaspersTue/SurgeNetSeg/resolve/main/ritm-surgenetseg-50k.pth?download=true"
```

You can also download these files directly from the Hugging Face project
page:\
https://huggingface.co/TimJaspersTue/SurgeNetSeg

Place the downloaded `.pth` files in a `gui/weights/` directory (or another
preferred location and point your scripts to that path).

------------------------------------------------------------------------

## Quick Start

### Interactive Demo

Start the interactive demo with:

``` bash
python gui.py --video examples/example.mp4
```

[See more instructions here](gui/INTERACTIVE.md).


## Citation

``` bibtex
@inproceedings{...
}
```

## References

-   The GUI tools used in this project are adapted from the excellent
    work in [Cutie](https://github.com/hkchengrex/Cutie) and
    [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation).
    We would like to express our appreciation for their open-source
    contributions, which form the foundation for the interactive
    segmentation components of this project. Portions of our code are
    derived from or inspired by these repositories and are used in
    accordance with their respective licenses. For detailed instructions
    and additional capabilities, please refer to their original
    repositories.
