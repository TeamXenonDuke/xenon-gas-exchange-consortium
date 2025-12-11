# <sup>129</sup>Xe gas exchange imaging pipeline

The xenon gas exchange pipeline, developed at the [Driehuys Lab](https://sites.duke.edu/driehuyslab/), processes raw <sup>129</sup>Xe MRI data and produces a summary report to analyze the functionality of the human lung. This README presents the installation and basic usage the pipeline. Before moving to the installation process, download or clone this repository in your computer.

## Table of contents:

1. [Setup](#setup)

2. [Installation](#installation)

3. [Usage](#usage)

4. [Acknowledgments](#acknowledgements)

5. [How to Cite](#howtocite)

## Setup

The xenon gas exchange pipeline is a cross system_vendor program that works on Windows, Mac and Linux system. At least 8GB of RAM is required to run this pipeline. Windows users can install Windows Subsystem for Linux (WSL) or install Ubuntu as dual boot/in the virtual box. The details of WSL installation can be seen in Section 1.1. Warning: run time in WSL can be slower compare to Linux or Mac system.

### 1.1. Windows Subsystem for Linux

Windows Subsystem for Linux installation process can seem overwhelming, espcially following the procedure in the microsoft [documentation](https://docs.microsoft.com/en-us/windows/wsl/install-win10). However a short Youtube video can make the install process much easier. One good example YouTube instruction can be seen [here](https://www.youtube.com/watch?v=X-DHaQLrBi8&t=385s&ab_channel=ProgrammingKnowledge2ProgrammingKnowledge2). Note: If the YouTube link is broken, please search in YouTube.

### 1.2. Xcode and Homebrew for Mac

First, open the terminal and install Xcode Command Line Tools using this command:

```bash
xcode-select --install
```

If homebrew is not already installed, it can be installed using this command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Don't forget to add homebrew to your path. Check if homebrew installed correctly, writing `which brew`. The details of homebrew can be seen [here](https://brew.sh/).

## Installation

This program is written in Python. Some of the code is written in C language. We will begin with Python installation.

### 2.1. Python Installation

First step of the installation process is to install python. This gas exchange pipeline works with Python 3.8.8 in its current version. In order to install necessary Python Libraries, Python 3.8.8 version is required. To create a virtual environment, a 'conda' distribution is required. If you don't have conda distribution installed into your machine, you can install one downloading 'Anaconda' or 'Miniconda'. You can download the 'Anaconda Distribution' from this [link](https://www.anaconda.com/products/individual), or 'Miniconda' from this [link](https://docs.conda.io/en/latest/miniconda.html). Here, command line installation procedure has been presented. So, Mac user can download the Command Line Installer.

#### 2.1.1. Conda Installation on Intel Based Mac or Linux systems:

Now, open the terminal. You need to write a command to install Anaconda/Miniconda. Command to install the Anaconda or Miniconda is:

```bash
bash ~/path/filename
```

Example: If your downloaded Anaconda file is in the "Downloads" folder and the file name is "Anaconda3-2020.11-Linux-x86_64.sh", write the following in the terminal:

```bash
bash ~/Downloads/Anaconda3-2020.11-Linux-x86_64.sh
```

Here, path = Downloads and filename = Anaconda3-2020.11-Linux-x86_64.sh

Press "enter" and reply "yes" to agree to the license agreements. After completing the installation process, close and reopen the terminal. You can verify if you have `conda` now by typing `which conda` in your terminal.

If you don't see 'conda' directory after verifing, you can review the details of [Anconda](https://docs.anaconda.com/anaconda/install/linux/) or [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) installation.

#### 2.1.2. Conda Installation on Apple Silicon Based Mac systems:

For full use of this pipeline with Apple silicon based Mac systems, Conda should be installed with miniforge. Install miniforge using Homebrew using the following command:

```bash
brew install miniforge
```

More information on miniforge and alternative installation methods can be found at the [miniforge GitHub](https://github.com/conda-forge/miniforge).

#### 2.1.3. Conda Installtion on Windows Subsystem for Linux(WSL):

WSL users need to install `Anaconda` or 'Miniconda' for Linux inside the WSL shell. Change your current directory to where you have downloaded your Anaconda or Miniconda installation file (.sh file). Then run the:

```bash
bash filename
```

You can verify if you have `conda` now by typing `which conda` in your terminal.

### 2.2. Virtual Environment Creation and Python Package Installation

#### 2.2.1. Create Virtual Environment

To create a virtual environment using `conda` execute the command in the terminal:

```bash
conda create --name XeGas python=3.8.8
```

Here, XeGas is the the given name, but any name can be given.

To activate the environment, execute the command

```bash
conda activate XeGas
```

#### 2.2.2. Install Required Packages

We will be using pip to install the required packages. First update the pip using:

```bash
pip install --upgrade pip
```

Now install a c compiler. Here we will install gcc compiler.

##### Linux and WSL users:

Get updates:

```bash
sudo apt-get update
```

Install gcc executing this command:

```bash
sudo apt install gcc
```

##### Mac Users:

Install gcc:

```bash
brew install gcc
```

##### Installing Packages in the Virtual Environment:

Now we are ready to install necessary packages. Packages must be installed inside the virtual conda environment. The list of packages are in the `setup/requirements.txt` file. In the terminal, if you are not in the main program directory, change the directory using the `cd` command. To install the required packages, execute the command:

```bash
pip install -r setup/requirements.txt
```

To confirm that correct packages are installed, execute the command

```
pip list
```

and verify that the packages in the virtual environment agree with that in the `requirements.txt` file.

Additionally, execute the following command in your virtual conda environment:

```
conda install -c conda-forge weasyprint
```

##### Install Packages in your Native Computer:

For Linux and WSL:

```
sudo apt install poppler-utils
```

For Mac:

```
brew install poppler
```

### 2.3. Compilation and Download Necessary tools

#### 2.3.1. For Segmentation: Downloading the h5 models for machine learning

Check the `models/weights` folder if two `.h5` files are not available there, download `model_ANATOMY_UTE.h5` and `model_ANATOMY_VEN.h5` from this [link](https://drive.google.com/drive/folders/1gcwT14_6Tl_2zkLZ_MHsm-pAYHXWtVOA?usp=sharing) and place it in the `models/weights` folder in your main program directory.

#### 2.3.2. For Registration: Compiling ANTs

Compiling ANTs require to install git, cmake, g++, zlib. Following commands will install these packages.

#### Linux and Windows Subsystem For Linux(WSL) Users

Execute following commands on your terminal:

```bash
sudo apt-get -y install git
sudo apt-get -y install cmake
sudo apt install g++
sudo apt-get -y install zlib1g-dev
```

#### Mac Users:

Check if you have git, cmake, g++ using `which git`, `which cmake`, `which g++`

If you don't have any of these, execute the following commands:

```bash
brew install git
brew install cmake
brew install g++
```

Now we are ready to perform SuperBuild. Execute the following commands on your terminal.

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

Warning: This might take a while.

After sucessful ANTs SuperBuild, you will have to move 'antApplyTransforms', 'antsRegistration' and 'N4BisaFieldCorrection' files to `xenon-gas-exchange-consortium/bin/`. Now we are ready to process MRI scan of human lung.

Note: If necesary, the details of ANTs Compilation can be seen [here](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS)

## Usage

### 3.1. General usage

#### 3.1.1 Accepted file inputs

The pipeline accepts Siemens twix (.dat) or ISMRMRD (.h5) files for standard proton UTE, 1-point Dixon, and (optionally) calibration scans. Alternatively, if a subject scan has already been processed through the pipeline and you wish to reprocess the previously constructed images, you can run the pipeline on the subject's .mat file. ISMRMRD files must be named and formatted according to the <sup>129</sup>Xe MRI clinical trials consortium specifications: [https://github.com/Xe-MRI-CTC/xemrd-specification](https://github.com/Xe-MRI-CTC/xemrd-specification)

More information on consortium protocol for the proton UTE, 1-point Dixon, and calibration scans can be found in the following reference:
>Niedbalski, PJ, Hall, CS, Castro, M, et al. Protocols for multi-site trials using hyperpolarized 129Xe MRI for imaging of ventilation, alveolar-airspace size, and gas exchange: A position paper from the 129Xe MRI clinical trials consortium. Magn Reson Med. 2021; 86: 2966–2986. https://doi.org/10.1002/mrm.28985

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

### 3.3  Manual Segmentation Workflow

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

