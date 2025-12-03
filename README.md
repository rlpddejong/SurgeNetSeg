# 🚀 SurgeNetSeg: Clip-Level Anatomy Segmentation for Minimally Invasive Surgery

![Figure](figures/example_figure.png)

![Demo](figures/SurgeNetSeg_Labeling_Tool_Demo.gif)

## 🛠️ Installation

Tested on Linux, Windows, and MAC OS.

**Prerequisites:**

-   Python 3.8+
-   PyTorch 1.12+ and corresponding torchvision (preferably with CUDA
    GPU support for quick label propagation on Linux and Windows).

**Clone our repository:**

``` bash
git clone https://github.com/rlpddejong/SurgeNetSeg.git
```

**Install with pip:**

``` bash
cd SurgeNetSeg
pip install -e .
```


## ⚡ Quick Start: Video labeling tool

**Opening the tool**

Start the interactive labeling tool with:

``` bash
python gui.py --video examples/example.mp4
```

This automatically downloads the model weights from huggingface into the `gui/weights` folder. And next, it extracts frames from an example video from the [Cholec80 dataset](https://camma.unistra.fr/datasets/) located in the `examples` folder. Then the GUI will open and be ready for labeling. 

**Using the tool**

See [TIPS](gui/TIPS.md) for some tips on using the tool. These are also shown at the top right inside the labeling tool.

The stored masks will be placed inside the `workspace` folder. After trying the demo, it is also possible to put your frames in this folder directly to label data. If a `masks` folder exists in the workspace, we will use that to initialize the mask. That way, you can continue annotation from an interrupted run as long as the same workspace is used.

**Modifying configurations and classes**

There are additional configurations that you can modify in [gui_config](gui/cutie/config/gui_config.yaml).

The classes can be modified inside [palette.py](gui/cutie/utils/palette.py). Here you choose the number of classes, the name, and its color. *Be aware that more classes result in slower temporal mask propagation!*


## 📦 Pretrained Models

The pretrained model weights should be downloaded automatically upon first run. In case this does not work, they can be downloaded here: 
-   **cutie-surgenetseg-50k.pth**\
    https://huggingface.co/TimJaspersTue/SurgeNetSeg/resolve/main/cutie-surgenetseg-50k.pth?download=true
-   **ritm-surgenetseg-50k.pth**\
    https://huggingface.co/TimJaspersTue/SurgeNetSeg/resolve/main/ritm-surgenetseg-50k.pth?download=true

Place the downloaded `.pth` files in a `gui/weights/` directory.

## 📚 Citation

To be added.

``` bibtex
@inproceedings{...
}
```

## 🙏 Acknowledgments

The GUI tools used in this project are adapted from the excellent
    work in [Cutie](https://github.com/hkchengrex/Cutie) and
    [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation).
    We would like to express our appreciation for their open-source
    contributions, which form the foundation for the interactive
    segmentation components of this project. Portions of our code are
    derived from or inspired by these repositories and are used in
    accordance with their respective licenses. For detailed instructions
    and additional capabilities, please refer to their original
    repositories.
