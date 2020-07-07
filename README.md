# `fastSF`

`fastSF` is an open source hybrid parallel C++ code to compute structure functions for a given velocity or scalar field.

## Getting the Source Code

`fastSF` is hosted on GitHub. You can download the source code from the following link:

https://github.com/ShubhadeepSadhukhan1993/fastSF

## Installing `fastSF`

### Required Libraries

The following libraries are required for installing and running fastSF:

1. [`CMake`](https://cmake.org/download/) (Version 3.16.4)
2. [`Blitz++`](https://github.com/blitzpp/blitz) (Version 1.0.2)
3. [`YAML-cpp`](http://turbulencehub.org/index.php/codes/tarang/installing-tarang/)(Version 0.3.0) 
4. [`MPICH`](https://www.mpich.org/downloads/)(Version 3.3.2)
5. [`HDF5`](http://turbulencehub.org/index.php/codes/tarang/installing-tarang/)(Version 1.8.20)
6. [`H5SI`](https://github.com/anandogc/h5si)(Version 1.1.1)

Clicking on one of the above libraries will redirect you to the webpage that provides instructions for downloading and installation of that library.

IMPORTANT: 

* Note that `fastSF` is not compatible with higher versions `YAML-cpp`(> 0.3.0). 

* The library `H5SI`has been successfully tested with `HDF5` version 1.8.20. However, `H5SI` may not be compatible with higher versions of `HDF5`. Thus, the users are strongly recommended to install `HDF5` 1.8.20.  


###  Compiling instruction

After downloading `fastSF`, change into `fastSF/src` directory and run the command `make` in the terminal. An executable named `fastSF.out` will be created inside the `fastSF/src` folder.

## Testing `fastSF`
`fastSF` offers an automated testing process to validate the code. The relevant test scripts can be found in the `tests/` folder of the code. To execute the tesing process, change into `fastSF` and run the command 

`bash runTest.sh`. 

The code then runs four test cases; these are as follows. 

* In the first case, the code will generate a 2D velocity field given by **u** = [*x, z*], and compute the structure functions for the given field. For this case, the longitudinal structure functions should equal *l<sup>q</sup>*. 

* In the second case, the code will generate a 2D scalar field given by *T = x + z*, and compute the structure functions for the given field. For this case, the structure functions should equal *(l<sub>x</sub> + l<sub>z</sub>)<sup>q</sup>*.

* In one case, the code will generate a 3D velocity field given by **u** = [*x, y, z*], and compute the structure functions for the given field. For this case, the longitudinal structure functions should equal *l<sup>q</sup>*. 

* In the second case, the code will generate a 3D scalar field given by *T = x + y + z*, and compute the structure functions for the given field. For this case, the structure functions should equal *(l<sub>x</sub> + l<sub>y</sub> + l<sub>z</sub>)<sup>q</sup>*.

For the above cases, `fastSF` will compare the computed structure functions with the analytical results. If the percentage difference between the two values is less than 10<sup>-10</sup>, the code is deemed to have passed. 

Finally, for visualization purpose, the python script `test/test.py` is invoked. This script generates the plots of the second and third-order longitudinal structure functions versus *l*, and the density plots of the computed second-order scalar structure functions and *(l<sub>x</sub> + l<sub>z</sub>)<sup>2</sup>*. For the 3D scalar field, the density plots of the computed second-order scalar structure functions and *(l<sub>x</sub> + l<sub>y</sub> + l<sub>z</sub>)<sup>2</sup>* These plots demonstrate that the structure functions are computed accurately. Note that the following python modules are needed to run the test script successfully:

1. `h5py`
2. `numpy`
3. `matplotlib`


## Detailed instruction for running `fastSF`

This section provides a detailed procedure to execute `fastSF` for a given velocity or scalar field.

`fastSF` has a folder named `in`. This folder contains the input field files in `hdf5` format, and a parameters file named `para.yaml`. You need to provide the required input parameters in this file. The details of the entries are as follows:


### i) `para.yaml` details


#### `program: scalar_switch`

You can enter `true` or `false`

`true`: Calculate the structure function for a given scalar field. 

`false`: Calculate the structure function for a given velocity field. 

#### `program: 2D_switch`

You can enter `true` or `false`.

`true`: Calculate the structure function for two dimensional fields. 

`false`: Calculate the structure function for three dimensional fields.

#### `program: Only_logitudinal`

This entry is for structure functions for velocity  fields only. You can enter `true` or `false`.

`true`: Compute only the longitudinal structure function.

`false`: Compute both longitudinal and transverse structure functions.


#### `grid: Nx, Ny, Nz`

The number of points along *x*, *y*, and *z* direction respectively of the  grid. Valid for both the vector and scalar fields. 
For two dimensional fields you need to provide `Nx` and `Nz`.

Only integer values will be accepted.

#### `domain_dimension: Lx, Ly, Lz`

Length of the cubical box along *x*, *y*, and *z* direction respectively. 
For two dimensional fields, you need to provide `Lx` and `Lz`. 


#### `structure_function: q1, q2`

The lower and the upper limit of the order of the structure functions to be computed.

#### `test: test_switch`

You can enter `true` or `false`

`true`: For running in test mode. Idealized velocity and scalar fields are generated internally by the code. Computed structure functions are compared with analytical results. The code is PASSED if the percentage difference between the two results is less than `1e-10`.

`false`: The "regular" mode, in which the code reads the fields from the hdf5 files in the `in` folder.

### ii) Files Required:

All the files storing the input fields should be inside the `in` folder.

#### For two dimensional fields

For vector field, two files named as `U.V1r.h5` and `U.V3r.h5` are required. Each file has one dataset.

For scalar field, one file named as `T.Fr.h5` is required. Each file has one dataset.

Size of the array stored in these files should be (`Nx,Nz`). 

*Important:* Dataset name should be the same as the file name. For example, the dataset inside the file `U.V1r.h5` should be named `U.V1r`.

#### For three dimensional fields

For vector field, three files named as `U.V1r.h5`, `U.V2r.h5`, and `U.V3r.h5` are required. Each file has one dataset.

For scalar field, one file named as `T.Fr.h5` is required. Each file has one dataset.

Size of the array stored in these files should be (`Nx, Ny, Nz`). 

*Important:* Dataset name should be the same as the file name. For example, the dataset inside the file `U.V1r.h5` should be named `U.V1r`.


### iii) Running Instructions
Open the terminal change into `fastSF/in` folder. Open `para.yaml` to set all the parameters. Keep all the required files compatible with the parameter file. Now, move out of the `in` folder run the command

`mpirun -np [number of MPI processors] src/fastSF.out [number of processors in x direction]`

If the number of processors in x direction is not provided, the code will take it to be 1.

### iii) Output Information

**Velocity structure functions**:

The logitudinal and transverse structure functions of order `q` are stored in the files `SF_Grid_pll`+`q`+`.h5` and `SF_Grid_perp`+`q`+`.h5` respectively as two/three dimensional arrays for two/three dimensional input fields. 


**Scalar structure functions**:

The structure functions of order `q` are stored in the files `SF_Grid_pll`+`q`+`.h5` as two/three dimensional arrays for two/three dimensional input fields. 

## Documentation

The documentation can be found in `fastSF/docs/index.html`

## Contributions and Bug Reports

We welcome contributions to this project. If you wish to contribute, please create a branch with a [pull request](https://github.com/ShubhadeepSadhukhan1993/fastSF/pulls) and the changes can be discussed there.

If you find a bug in the or errors in the documentation, please open a [new issue](https://github.com/ShubhadeepSadhukhan1993/fastSF/issues/new) in the Github repository and report the bug or the error. Please provide sufficient information for the bug to be reproduced.  

## License

`fastSF` is released under the terms of BSD New License.

