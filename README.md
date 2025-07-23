# SurgeNetSeg: A large-scale Dataset for Surgical Semantic Segmentation

![Figure](figures/example_figure.png)

## Installation

Tested on Ubuntu only.

**Prerequisites:**

- Python 3.8+
- PyTorch 1.12+ and corresponding torchvision

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
python python interactive_demo.py --num_objects 14 --video P0024video1_01-38-43_0148075_survey.mp4  
```

[See more instructions here](docs/INTERACTIVE.md).


## Citation

```bibtex
@inproceedings{...
}
```

## References

- The GUI tools are adapted from [Cutie](https://github.com/hkchengrex/Cutie) and [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation). This part of the code follows their individual licenses. For more information on fine-tuning these models, please refer to their original repositories.


