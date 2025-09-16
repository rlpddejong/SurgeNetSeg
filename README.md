# SurgeNetSeg: A large-scale Dataset for Surgical Semantic Segmentation

![Figure](figures/example_figure.png)

## Installation

Tested on Ubuntu only.

**Prerequisites:**

- Python 3.8+
- PyTorch 1.12+ and corresponding torchvision (preferably with cuda GPU support for quick label propagation)

**Clone our repository:**

```bash
git clone https://github.com/rlpddejong/SurgeNetSeg.git
```

**Install with pip:**

```bash
cd SurgeNetSeg
pip install -e .
```


## Quick Start

### Interactive Demo

Start the interactive demo with:

```bash
python gui.py --video vid_folder_name
```

[See more instructions here](gui/INTERACTIVE.md).


## Citation

```bibtex
@inproceedings{...
}
```

## References

- The GUI tools used in this project are adapted from the excellent work in [Cutie](https://github.com/hkchengrex/Cutie) and [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation). We would like to express our appreciation for their open-source contributions, which form the foundation for the interactive segmentation components of this project. Portions of our code are derived from or inspired by these repositories and are used in accordance with their respective licenses. For detailed instructions and additional capabilities, please refer to their original repositories.