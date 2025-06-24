# <sup>129</sup>Xe gas exchange imaging pipeline

The xenon gas exchange pipeline, developed at the [Driehuys Lab](https://sites.duke.edu/driehuyslab/), processes raw <sup>129</sup>Xe MRI data and produces a summary report to analyze the functionality of the human lung. This README presents the setup/installation process and basic usage of the pipeline. Before moving to the installation process, download or clone this repository to your computer.

## Table of contents:

1. [Setup](#setup)

2. [Installation](#installation)

3. [Usage](#usage)

4. [Acknowledgments](#acknowledgements)

5. [How to Cite](#howtocite)

## Setup

The xenon gas exchange pipeline is a cross system-vendor program that works on Windows (WSL), Mac, and Linux system. At least 8GB of RAM is required to run this pipeline.

Windows users must install Windows Subsystem for Linux (WSL) or install Ubuntu as dual boot/in the virtual box.

Mac users must install Xcode Command Line Tools and Homebrew.

WARNING: run time in WSL can be slower compared to Linux or Mac systems.

### 1.1. Windows Subsystem for Linux

The install process for WSL using Ubuntu can be found [here](https://www.youtube.com/watch?v=X-DHaQLrBi8&t=385s&ab_channel=ProgrammingKnowledge2ProgrammingKnowledge2). Note: If the link is broken, please search "How to Install Ubuntu on Windows 10 (WSL)" on YouTube.

All terminal commands for Windows users should be through WSL on Ubuntu. You must use the Ubuntu terminal for setup, installation, and when using the pipeline.

Further information for the Windows Subsystem for Linux installation process can be found in the [Microsoft documentation](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

### 1.2. Xcode and Homebrew for Mac

Open the terminal and install Xcode Command Line Tools using the following command:

```bash
xcode-select --install
```

If Homebrew is not already installed, install it using the following command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Check if Homebrew installed correctly to your PATH, with `which brew`. The details of Homebrew can be found [here](https://brew.sh/).

## Installation

### 2.1. Python and Conda Installation

The pipeline requires Python 3.8.8. We will setup and run the program using a dedicated virtual environment. To create a virtual environment, a `conda` distribution is required. You can install a conda distribution by downloading 'Anaconda' or 'Miniconda'. You can download 'Anaconda' from this [link](https://www.anaconda.com/products/individual), or 'Miniconda' from this [link](https://docs.conda.io/en/latest/miniconda.html). (Note: Windows users should download the Linux version since they will be using WSL)

Select Conda installation for:

[2.1.1. Linux or Intel Based Mac](#211-conda-installation-on-intel-based-mac-or-linux-systems)

[2.1.2. Apple Silicon Based Mac](#212-conda-installation-on-apple-silicon-based-mac-systems)

[2.1.3. Windows Subsystem for Linux (WSL)](#213-conda-installation-on-windows-subsystem-for-linux-wsl)

#### 2.1.1. Conda Installation on Linux or Intel Based Mac systems:

Open the terminal. Input the following command to install the Anaconda or Miniconda distribution:

```bash
bash ~/<path>/<filename>
```

`<path>` is the absolute path to where the Anaconda file is located. `<filename>` is the name of the installation file.

Example: If your downloaded Anaconda file is in the "Downloads" folder and the file name is "Anaconda3-2020.11-Linux-x86_64.sh", write the following in the terminal:

```bash
bash ~/Downloads/Anaconda3-2020.11-Linux-x86_64.sh
```

Press "enter" and reply "yes" to agree to the license agreements. After completing the installation process, close and re-open the terminal. You can verify if conda was installed correctly with `which conda`.

If you do not see the 'conda' folder after installation, you can review the details of [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) or [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) installation.

#### 2.1.2. Conda Installation on Apple Silicon Based Mac systems:

On Apple silicon based Mac systems, Conda must be installed with miniforge. Install miniforge using Homebrew with the following command:

```bash
brew install miniforge
```

More information on miniforge and alternative installation methods can be found at the [miniforge GitHub](https://github.com/conda-forge/miniforge).

#### 2.1.3. Conda Installation on Windows Subsystem for Linux (WSL):

WSL users must install 'Anaconda' or 'Miniconda' for Linux inside the WSL shell. `cd` to the folder where you have downloaded your Anaconda or Miniconda installation file (.sh file). Then run:

```bash
bash <filename>
```

`<filename>` is the specific name of the installation file (e.g. `bash Anaconda3-2024.10-1-Linux-x86_64`).

You can verify if you have conda by typing `which conda` in your terminal.

### Installation Using `setup.sh`

Steps 2.2 and 2.3 of installation can be completed automatically using the bash script `setup.sh` in the pipeline folder. If you wish to complete the process manually, skip to [Step 2.2](#22-virtual-environment-creation-and-package-installation).

`cd` to the main pipeline folder (`xenon-gas-exchange-consortium`) before running the next commands!

Run the following commands to automate steps 2.2 and 2.3. The first command will allow you to run `setup.sh` on your terminal.

```bash
chmod u+x ./setup.sh
```

The second command will run the script and record all output and errors to `setup.log`:

```bash
./setup.sh 2>&1 | tee setup.log
```

The terminal will prompt you to complete steps that require some user input.

1. The terminal will ask you to enter a name for the conda virtual environment. `XeGas` is the recommended default.

   a. If you have previously created an environment and want to use it for setup/installation, enter the name for it. The terminal will ask if you would like to create a new one with the same name or use the existing environment for setup.

2. You will be asked to check the list of packages installed by `setup.sh` to make sure they are all there. Check over the list (shown by `pip list`) to verify if they include all packages in `setup/requirements.txt`.

3. You will need to download the files `model_ANATOMY_UTE.h5` and `model_ANATOMY_VEN.h5` from this [link](https://drive.google.com/drive/folders/1gcwT14_6Tl_2zkLZ_MHsm-pAYHXWtVOA?usp=sharing) and place them in `models/weights`.

4. You will be prompted to enter the number of threads you wish to use during the SuperBuild step. 4 is recommended. Using more can cause your OS to force cancel during build/installation. More threads will make the process faster. If you keep getting failures, switch to 1 thread.

If you are having issues with the SuperBuild failing (build and install step), please see [Resolving Issues with `setup.sh` SuperBuild](#resolving-issues-with-setupsh-superbuild).

Setup and installation will be complete and the pipeline will be ready to use once `setup.sh` is ran successfully.

Remember to use the following command before using the pipeline:

```bash
conda activate <name_of_conda_environment>
```

The next steps will be for manual installation. Skip to [Usage](#usage) if you are finished with setup and installation.

#### Resolving Issues with `setup.sh` SuperBuild

You can rerun `setup.sh` to skip to the build and install step if there is a failure during the SuperBuild. You should have all required packages installed to your conda environment and native computer by the time the SuperBuild starts, so you can skip over that part of `setup.sh` and go straight to the build or install steps.

If there was a failure during the build step, enter the following command to skip to the build step (Note: Repeated failures may mean you are using too many CPU cores or RAM. You may need to run the build step with only 1 thread!):

```bash
./setup.sh build-only 2>&1 | tee setup.log
```

If you made it passed the build step successfully but had a failure during the install step, enter the following command to skip to the install step:

```bash
./setup.sh install-only 2>&1 | tee setup.log
```

Setup and installation will be complete and the pipeline will be ready to use once `setup.sh` is ran successfully.

Remember to use the following command before using the pipeline:

```bash
conda activate <name_of_conda_environment>
```

The next steps will be for manual installation. Skip to [Usage](#usage) if you are finished with setup and installation.

### 2.2. Virtual Environment Creation and Package Installation

#### 2.2.1. Create Virtual Environment

To create a virtual environment using `conda` execute the following command in the terminal. "XeGas" is the recommended name, but you may use a different name:

```bash
conda create --name XeGas python=3.8.8
```

To activate the environment, execute the following commands. Replace "XeGas" with the name you previously chose if needed:

```bash
conda init
conda activate XeGas
```

#### 2.2.2. Install Required Packages

##### Install/Update pip

We will be using pip to install the required packages. Update pip using:

```bash
pip install --upgrade pip
```

##### Install C Compiler

Next we will install gcc compiler to compile C code using sudo (Linux/WSL) or brew (Mac).

Linux and WSL users:

```bash
sudo apt-get update
sudo apt install gcc
```

Mac Users:

```bash
brew install gcc
```

##### Install Packages in your Virtual Environment

Packages must be installed inside the virtual conda environment. Make sure your VE is activated using `conda activate XeGas`.

`cd` to the main pipeline folder (`xenon-gas-exchange-consortium`) before running the next command!

The list of packages are in `setup/requirements.txt`. To install the required packages, run the following command:

```bash
pip install -r setup/requirements.txt
```

To confirm that the correct packages are installed, execute the following command:

```bash
pip list
```

Verify that the packages in the virtual environment include the packages in the `requirements.txt` file.

##### Install Packages in your Native Computer

Lastly, install the following packages directly to your computer using sudo or brew.

Linux and WSL Users:

```bash
sudo apt install wkhtmltopdf
sudo apt install poppler-utils
```

Mac Users:

```bash
brew install wkhtmltopdf
brew install poppler
```

### 2.3. Download Necessary Tools and Compile ANTs

#### 2.3.1. Segmentation: Downloading the .h5 models for machine learning

Download `model_ANATOMY_UTE.h5` and `model_ANATOMY_VEN.h5` from this [link](https://drive.google.com/drive/folders/1gcwT14_6Tl_2zkLZ_MHsm-pAYHXWtVOA?usp=sharing) and place it in the `models/weights` folder in your main pipeline folder (`xenon-gas-exchange-consortium`).

#### 2.3.2. Registration: Compiling ANTs

Compiling ANTs requires `git`, `cmake`, `g++`, and `zlib`. The following commands will install these packages.

Linux and WSL Users:

```bash
sudo apt-get -y install git
sudo apt-get -y install cmake
sudo apt install g++
sudo apt-get -y install zlib1g-dev
```

Mac Users:

Check if you have git, cmake, g++ using `which git`, `which cmake`, `which g++`

If you do not have any of these, execute the following commands:

```bash
brew install git
brew install cmake
brew install g++
```

##### Compile ANTs

We are ready to perform SuperBuild. Execute the following commands on your terminal.

Do NOT run commands using an integrated terminal in a code editor like VSCode. The code editor will significantly slow down the build. Run the following commands in a separate, independent terminal.
Close any applications you are not using.

WARNING: This may take a while.

```bash
workingDir=${PWD}
git clone https://github.com/ANTsX/ANTs.git
mkdir build install
cd build
cmake \
    -DCMAKE_INSTALL_PREFIX=${workingDir}/install \
    ../ANTs 2>&1 | tee cmake.log
make -j 4 2>&1 | tee build.log
cd ANTS-build
make install 2>&1 | tee install.log
```

`make -j 4 2>&1 | tee build.log` may fail after running. Re-run the command using less threads if it fails repeatedly (e.g. `make -j 1 2>&1 | tee build.log`). Make sure to `cd` to the `build` folder before running.

`make install 2>&1 | tee install.log` may also fail and need to be re-ran. Make sure to `cd` to the `ANTS-build` folder before running.

Once SuperBuild is finished, move the `antsApplyTransforms`, `antsRegistration`, and `n4BiasFieldCorrection` files to the bin folder (`xenon-gas-exchange-consortium/bin/`) with the following command:

```bash
mv ./Examples/antsApplyTransforms ./Examples/antsRegistration ./Examples/N4BiasFieldCorrection ../../bin
```

Note: The details of ANTs Compilation can be found [here](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS).

## Usage

### 3.1. General usage

#### 3.1.1 Accepted file inputs

The pipeline accepts Siemens twix (.dat) or ISMRMRD (.h5) files for standard proton UTE, 1-point Dixon, and (optionally) calibration scans. Alternatively, if a subject scan has already been processed through the pipeline and you wish to reprocess the previously constructed images, you can run the pipeline on the subject's .mat file. ISMRMRD files must be named and formatted according to the <sup>129</sup>Xe MRI clinical trials consortium specifications: [https://github.com/Xe-MRI-CTC/xemrd-specification](https://github.com/Xe-MRI-CTC/xemrd-specification)

More information on consortium protocol for the proton UTE, 1-point Dixon, and calibration scans can be found in the following reference:

> Niedbalski, PJ, Hall, CS, Castro, M, et al. Protocols for multi-site trials using hyperpolarized 129Xe MRI for imaging of ventilation, alveolar-airspace size, and gas exchange: A position paper from the 129Xe MRI clinical trials consortium. Magn Reson Med. 2021; 86: 2966–2986. https://doi.org/10.1002/mrm.28985

#### 3.1.2 Config file

All subject information and processing parameters are specified in a subject-specific configuration file. Default configuration settings are defined in `config/base_config.py`. The defaults are inhereted by subject-specific config files, unless overriden.
<br />
<br />`config/demo_config_basic.py` shows examples of basic config settings that you will usually want to change for each subject scan.

- `data_dir`: Directory containing Dixon, proton, and (optionally) calibration scan files or .mat file. This is where output files will be saved.
- `subject_id`: Subject ID number that will be used to label output files
- `rbc_m_ratio`: RBC to membrane signal ratio for Dixon decomposition. If not set in config file, a calibartion scan file is required from which the ratio will be calculated.
- `segmentation_key`: Defines what kind of segmentation to use. Typically set to CNN_VENT for automatic segmentation of the gas image or MANUAL_VENT for manual segmentation of the gas image.
- `manual_seg_filepath`: Path of manual segmentation file, if MANUAL_VENT is chosen.

`config/demo_config_advanced.py` shows examples of advanced config settings that may commonly be modified for more specific cases. See `config/base_config.py` for all config settings that can be modified.

#### 3.1.3 Processing a subject

First, copy one of the demo config files or the base_config file, rename it, and modify configuration settings. In terminal, navigate to the main pipeline directory and activate the virtual environment you set up earlier:

```bash
conda activate XeGas
```

#### Running full pipeline with image reconstruction

Run the full pipeline with:

```bash
python main.py --config [path-to-config-file]
```

#### Running previously processed subject scan from .mat file

If a subject scan has already been processed through the pipeline and you wish to reprocess the previously constructed images, you can run the pipeline on the subject's .mat file with:

```bash
python main.py --config [path-to-config-file] --force_readin
```

### 3.2. Team Xenon Worflow for Duke Data Processing

Warning: this is the Team Xenon workflow only. Other users do not have to follow the exact procedures.

1. Create a new subject folder. This will typically have the format of `###-###` or `###-###X`.

2. Then log onto the `smb://duhsnas-pri/xenon_MRI_raw/` drive and enter the directory of interest corresponding to the recently scanned subject. Copy the files on your computer. Determine how many dixon scans are there (usually 1 or 2). If there is only 1, create a subfolder named `###-###` in your new subject folder and copy all twix files(should be at least three files: dixon, calibration, and BHUTE) into that subfolder.(NOTE: scan can be processed using only one dixon scan. In that case, only one dixon should be in the subfolder) If there are 2 dixons, create subfolders `###-###_s1` (for the first dixon scan) and `###-###_s2`(for the second dixon scan) and copy the twix files corresponding to the first dixon (cali, dixon, ute, and optionally dedicated ventilation) and copy the twix files corresponding the second dixon (cali, dixon, ute) into the other.

3. Process the spectroscopy using the MATLAB spectroscopy pipeline first. Instructions are on the repository ("Spectroscopy_Processing_Production").

4. Before running the gas exchange pipline, make sure you have the latest updates. You can do this by

   ```
   git pull
   ```

5. Create a new config file titled "[subject_id].py" in lower case by copying one of the demo config files. Then, edit the parameters like subject id and rbc/m ratio and save it. Run the pipeline.

6. Copy all the contents in the subject folder and paste it into `smb://duhsnas-pri/duhs_radiology/Private/TeamXenon/01_ClinicalOutput/Processed_Subjects`

7. Upload `.pdf` reports to Slack

### 3.3 Manual Segmentation Workflow

Note: The following steps are for correcting auto-generated gas image masks.

1. Ensure you have [version 3.8](https://sourceforge.net/projects/itk-snap/files/itk-snap/3.8.0/) of ITK-SNAP installed.

2. Open `gas_highreso.nii` in ITK-SNAP as Main Image.

3. Load `proton_reg.nii` as Additional Image and `mask_reg.nii` as Segmentation.

4. Set Display Layout to Axial View and Thumbnail Layout.

5. Correct the mask with the Paintbrush and save as `mask_reg_corrected.nii` when complete.

6. In your config file, set `segmenation_key` to MANUAL_VENT and `manual_seg_filepath` to the corrected mask filepath.

7. Reprocess subject as specified in 3.1.3.

## Acknowledgements:

Developers: Junlan Lu, Aryil Bechtel, Sakib Kabir, Suphachart Leewiwatwong, David Mummy.

Code inspired by: Ziyi Wang

Additional help: Isabelle Dummer, Joanna Nowakowska, Shuo Zhang

Please contact David Mummy (david.mummy@duke.edu) for any correspondence.

## How to Cite:

We appreciate being cited! Please click the "Cite This Repository" button under "About" on the repository landing page to get APA and BibTex citations. You can also just copy the following BibTex code into a plain text file and load it into your favorite citation manager:

@software{Lu_Duke_University_Xenon_2024,
author = {Lu, Junlan and Leewiwatwong, Suphachart and Bechtel, Ari and Kabir, Sakib and Wang, Ziyi},
month = jan,
title = {{“Duke University Xenon Gas Exchange Imaging Pipeline”}},
url = {https://github.com/TeamXenonDuke/xenon-gas-exchange-consortium},
version = {4.0},
year = {2024}
}
