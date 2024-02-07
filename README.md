# MultilayerPerceptron

This project is an implementation of a multilayer perceptron for classifying handwritten letters of the Latin alphabet. Using this application you can train the neural network, analyse the results and interact with the graphical user interface developed on the Qt library.

## Core functionality

- **Classification of handwritten letters:** Recognition of handwritten letters of the Latin alphabet.
- **Multilayering:** Possibility to customise from 2 to 5 hidden layers.
- **Sigmoidal activation function:** Application of sigmoidal function in hidden layers.
- **Training:** Application of the error back propagation method.
- **Image handling:** Loading and categorising BMP images.
- **Results visualisation:** Real-time display of results and error plots.
- **Cross-validation:** Cross-validation capability with selectable number of groups.
- **Saving and loading the model:** Saving weights to a file and loading them.
- **Image drawing:**Function for drawing two-colour square images.
- **Qt interface:** Nice and intuitive graphical interface.

## Model realisations

The project offers two implementations of a multilayer perceptron:

- **Graph model:** Neurons are represented as node objects connected to each other.
- **Matrix model:** All layers are represented as weight matrices for optimised data handling.

## Installation

Follow these steps to build and install the project:

1. **Clone the repository:**
    ```
    git clone [URL repository]
    ```

2. **Go to the project directory:**
    ```
    cd src
    ```

3. **Project installation:**
    ```
    make install
    ```

After executing these commands, the project will be built and installed
on your system in the folder`bin/Release`.

## Build the whole project

To build the entire project, including installation, testing, styling, code validation, code coverage report generation, and archive creation.
code validation, code coverage report generation, and archive creation,
perform the following steps:

1. **Go to the project directory:**
    ```
    cd src
    ```

2. **Project assembly:**
    ```
    make all
    ```

After executing this command, all the steps described will be executed.

## Delete

To delete an installed project, perform the following steps:

1. **Go to the project directory:**
    ```
    cd src
    ```

2. **Deleting a project:**
    ```
    make uninstall
    ```

After executing these commands the project will be removed from your system.

#### P.S. If the installation files were moved from /bin/ then deletion will not occur, you need to delete the files manually.


## Testing

Follow these steps to build and run tests on the project:

1. **Move to the project directory:**
    ```
    cd src
    ```

2. **Build and run tests:**
    ```
    make tests
    ```

After executing these commands, an executable file will be created in the `bin/Testing` folder, which contains all the tests of the project. 
file containing all the tests of the project.

## Code coverage report

Perform the following steps to generate a code coverage report for tests:

1. **Go to the project directory:**
    ```
    cd src
    ```

2. **Report generation:**
    ```
    make gcov_report
    ```

After executing these commands, a code coverage report will be generated in the `bin/Testing` folder.

## Build project archive

Follow the steps below to build the project archive:

1. **Go to the project directory:**
    ```
    cd src
    ```

2. **Assembly of the archive:**
    ```
    make dist
    ```

After executing this command, an archive will be created in the project folder,
containing all necessary files for building and running the project.

## Code Style Check

To perform code style checking according to Google Style, follow the steps below:

1. **Go to the project directory:**
    ```
    cd src
    ```

2. **Start style check:**
    ```
    make check
    ```

After executing this command, the entire project code will be style-checked.

## Cleanup of build files

To remove all temporary and build files created during compilation and testing, follow these steps:

1. **Go to the project directory:**
    ```
    cd src
    ```

2. **Clearing the project:**
    ```
    make clean
    ```

After executing this command, all temporary and assembly files will be deleted.


# Created by [stevenso](https://github.com/v3ssel) and [gabriela](https://github.com/TanyaPh) in 2024 in educational purpose.
