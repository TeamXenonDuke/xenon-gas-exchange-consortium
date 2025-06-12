#!/bin/bash
RED='\033[0;31m'
NC='\033[0m'

echo "It is recommended to run this script with the following command './setup.sh 2>&1 | tee setup.log' in order to log all standard output and error in case of failure."

# Should exit if there are 1. too many parameters, 2. an unknown parameter for p1, or 3. setup.sh was run in a directory that isn't the one that hosts the pipeline.
if (( $# > 1 )); then
    echo -e "${RED}Error: Too many parameters.${NC}"
    echo "Exiting..."
    exit 1
fi
if (( $# == 1 )) && !([[ "$1" == "build-only" ]] || [[ "$1" == "install-only" ]]); then
    echo -e "${RED}Error: Unknown parameter: '$1'.${NC}"
    echo "Exiting..."
    exit 1
fi
if [[ "$PWD" != "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" ]]; then
    echo -e "${RED}You are not in the correct directory. ${NC}Please cd to the pipeline directory before running setup.sh."
    echo "Exiting..."
    exit 1
fi

# If we are running setup.sh without the build-only or install-only parameter, we will execute the whole script.
if !([[ "$1" == "build-only" ]] || [[ "$1" == "install-only" ]]); then

    # Conda setup. Create a name for the VE
    read -r -p "Enter a name for your virtual environment or press enter for default name: 'XeGas'. " reply
    while true; do
        if [[ "$reply" == "" ]]; then
            ve_name="XeGas"
            break
        elif [[ $reply =~ ^[^[:space:]]+$ ]]; then
            ve_name=$reply
            break
        else
            echo -e "${RED}Error: No spaces are allowed in the VE name.${NC} Please enter another name."
        fi
        read -r reply
    done

    # Check if the inputted environment already exists. If so, ask user if they want to use that environment or create a new one.
    if conda env list | grep -w $ve_name >/dev/null; then
        read -r -n 1 -p "Virtual environment already exists with name '$ve_name'. Would you like to use this VE or create a new one? Enter 'y' to use the existing VE or 'n' to be prompted to create a new VE. " reply
        echo
        while true; do
            if [[ $reply =~ ^[Yy]$ ]]; then
                break
            elif [[ $reply =~ ^[Nn]$ ]]; then
                echo "Creating conda VE named '$ve_name'..."
                conda create --name $ve_name python=3.8.8
                break
            else 
                echo -e "${RED}Error: Invalid input. ${NC}Please enter 'y' when if you want to use the VE or 'n' if you want to create a new one with the same name. "
            fi
            read -r -n 1 reply
            echo
        done
    else  
        # Create conda environment with Python 3.8.8
        echo "Creating conda VE named '$ve_name'..."
        conda create --name $ve_name python=3.8.8
    fi

    # conda init and conda activate
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate $ve_name
    echo "Current env: $CONDA_DEFAULT_ENV"

    # pip install and upgrade
    pip install --upgrade pip

    # Install gcc to native computer
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt install gcc
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install gcc
    else 
        echo -e "${RED}Error: Incorrect OS. ${NC}Please use WSL, Linux, or MacOS."
        echo "Exiting..."
        exit 1
    fi

    # Install packages in requirements.txt to the VE
    pip install -r setup/requirements.txt

    # Prompt for user: confirm that all packages are installed
    pip list
    read -r -n 1 -p "Please check the above list to confirm that all packages in './setup/requirements.txt' are installed. Enter 'y' when complete. Enter 'n' if you want to exit setup. " reply
    echo
    while true; do
        if [[ $reply =~ ^[Yy]$ ]]; then
            break
        elif [[ $reply =~ ^[Nn]$ ]]; then
            echo "Exiting..."
            exit 1
        else
            echo -e "${RED}Invalid input. ${NC}Please enter 'y' when complete or 'n' if you want to exit setup. "
        fi
        read -r -n 1 reply
        echo
    done

    # Install remaining packages to native computer
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt install wkhtmltopdf
        sudo apt install poppler-utils
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install wkhtmltopdf
        brew install poppler
    else 
        echo -e "${RED}Error: Incorrect OS. ${NC}Please use WSL, Linux, or MacOS."
        echo "Exiting..."
        exit 1
    fi

    # Prompt for user: pause to move the .h5 files to the correct directory from the Google Drive and confirm.
    read -r -n 1 -p "Please move the .h5 files to './models/weights'. Enter 'y' when complete. Enter 'n' if you want to exit setup. " reply
    echo
    while true; do
        if [[ $reply =~ ^[Yy]$ ]]; then
            break
        elif [[ $reply =~ ^[Nn]$ ]]; then
            echo "Exiting..."
            exit 1
        else
            echo -e "${RED}Invalid input. ${NC}Please enter 'y' when complete or 'n' if you want to exit setup. "
        fi
        read -r -n 1 reply
        echo
    done

    # Install git, cmake, g++, zlib to native computer
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get -y install git
        sudo apt-get -y install cmake
        sudo apt install g++
        sudo apt-get -y install zlib1g-dev
        #sudo apt install ccache
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git
        brew install cmake
        brew install g++
        #brew install ccache
    else 
        echo -e "${RED}Error: Incorrect OS. ${NC}Please use WSL, Linux, or MacOS."
        echo "Exiting..."
        exit 1
    fi

    # Begin setup for SuperBuild. Clone ANTs and create build and install directories
    workingDir=${PWD}
    git clone https://github.com/ANTsX/ANTs.git
    mkdir build install
    # Second confirmation that we are in the correct pipeline directory before continuing.
    if [[ "$PWD/build" == "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/build" ]]; then
        cd "$PWD"/build
    else 
        echo "${RED}Error: You are not in the correct directory. ${NC}Please cd to the pipeline directory before running setup.sh."
        echo "Exiting..."
        exit 1
    fi

    # Run cmake
    cmake \
        -DCMAKE_INSTALL_PREFIX=${workingDir}/install \
        ../ANTs 2>&1 | tee cmake.log
        #-DCMAKE_C_COMPILER_LAUNCHER=ccache \
        #-DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
else
    # If the 'build-only' parameter is used, we will skip the above and go straight to the SuperBuild. We assume that ./setup.sh has already been run before.
    # 'build-only' should ONLY be used if there is a failure during the SuperBuild and install.

    # Prompt for the VE used in the original run of ./setup.sh
    read -r -p "Enter the name for the virtual environment you created previously for this pipeline (default name: 'XeGas'). " reply
    while true; do
        if conda env list | grep -w $reply >/dev/null; then
            ve_name=$reply
            break
        else
            echo -e "${RED}Error: virtual environment not found. ${NC}Please enter the correct name. If you previously ran 'setup.sh' and created a virtual environment but forgot its name, exit setup and enter 'conda env list' in the terminal. If you deleted the virtual environment or have not run 'setup.sh' yet, please exit and run './setup.sh' without the 'build-only' parameter."
            read -r -n 1 -p "Exit setup? [y/n] " response
            echo
            if [[ $response =~ ^[Yy]$ ]]; then
                echo "Exiting..."
                exit 1
            fi
        fi
        read -r -p "Enter the name for the virtual environment you created previously for this pipeline (default name: 'XeGas'). " reply
    done
    
    # conda init and conda activate
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate $ve_name
    echo "Current env: $CONDA_DEFAULT_ENV"

    # Make sure we are in the correct pipeline directory before continuing.
    if [[ "$PWD/build" == "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/build" ]]; then
        workingDir=${PWD}
        cd "$PWD"/build
    else 
        echo -e "${RED}You are not in the correct directory. ${NC}Please cd to the pipeline directory before running setup.sh."
        echo "Exiting..."
        exit 1
    fi
fi

# Check that we ran ./setup.sh or ./setup.sh build-only
if (( $# == 0 )) || [[ "$1" == "build-only" ]]; then
    # Warning and prompts to the user to make sure they understand that the SuperBuild and install step is about to begin.
    echo -e "${RED}Warning: The ANTs SuperBuild is about to start! ${NC}"
    read -r -n 1 -p "It is recommended to run the superbuild in an independent terminal application rather than in a code editor (e.g. VSCode). Please do NOT start any new applications or processes! Make sure your laptop is plugged in. Enter 'y' when you are ready to start the SuperBuild. Enter 'n' if you want to exit setup. " reply
    echo
    while true; do
        if [[ $reply =~ ^[Yy]$ ]]; then
            break
        elif [[ $reply =~ ^[Nn]$ ]]; then
            echo "Exiting..."
            exit 1
        else
            echo -e "${RED}Error: Invalid input. ${NC}Please enter 'y' when ready to start or 'n' to exit setup. "
        fi
        read -r -n 1 reply
        echo
    done
    echo -e "${RED}WARNING! This part will take a while. Please do NOT start any new applications or processes! Make sure your laptop is plugged in. If the SuperBuild fails, rerun setup.sh as './setup.sh build-only' in the terminal.${NC}"

    # Check for available processor cores in case user wants to use more than the recommended 4.
    if command -v nproc >/dev/null; then
        available_cores=$(($(nproc) * 3 / 4))
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        available_cores=$(($(sysctl -n hw.ncpu) * 3 / 4))
    else
        available_cores=4
    fi

    # Prompt for user: User selects the number of cores they wish to allocate for the build step (max: 3/4 of available cores). Recommended default is 4.
    read -r -p "You have $available_cores cores available for threads. Please enter the number of cores you would like to use for the superbuild (max: $available_cores) or press enter for  the recommended default (4 cores). " reply
    while true; do
        if [[ "$reply" == "" ]]; then
            CORES=4
            break
        elif [[ $reply =~ ^[0-9]+$ ]] && (( $reply <= $available_cores )); then
            CORES=$reply
            break
        else
            echo -e "${RED}Error: Input was above max ($available_cores) or was not a number.${NC}"
        fi
        read -r reply
    done

    # Build step
    echo "$CORES cores allocated for threads."
    make -j $CORES 2>&1 | tee build.log
fi

# Install step
echo "Install step for SuperBuild starting..."
echo -e "${RED}WARNING! This part will take a while. Please do NOT start any new applications or processes! Make sure your laptop is plugged in. If the install fails, rerun setup.sh as './setup.sh install-only' in the terminal.${NC}"
cd ANTS-build
make install 2>&1 | tee install.log

# Move antsApplyTransforms, antsRegistration, and N4BiasFieldCorrection to bin
mv ./Examples/antsApplyTransforms ./Examples/antsRegistration ./Examples/N4BiasFieldCorrection ../../bin
echo "Please check that 'antsApplyTransforms', 'antsRegistration', and 'N4BiasFieldCorrection' are located in the 'bin' directory."

# Exit.
echo "Setup complete. Exiting..."
exit 0