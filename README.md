## Instructions for reproducing the paper results

First, one should have the docker container up and running with all
dependencies installed. All open source software is automatically
downloaded, except for IBM ILOG CPLEX, which needs either a professional
or an academic license. The docker container and instructions can be found at
https://github.com/joao-bapdm/docker-spyro 

The scripts used for generating input files and running the simulation can be found in the `reproducibility_integer_linear_programming` directory.

The folders there are structured as follows:
```
2D
└───reference_models
│   │   single_inclusion.hdf5
│   │   double_inclusion.hdf5
|
└───simulation_scripts
│   │   fwd.py
│   │   fwd_multiscale.py
│   │   fwd_combine.py
│   │   fwd_cplex_multiscale.py
│
└───section_5.1
    │   run_single_inclusion.sh
    │   
    └───inputs
        |...
│
└───section_5.2
    │   run_single_inclusion.sh
    │   
    └───inputs
        |...
│
└───section_5.3
    │   run_double_inclusion.sh
    │   
    └───inputs
        |...
│
└───section_5.4
    │   run_multiscale.sh
    │   
    └───inputs
        |...
│
3D
│   fwi_ball_tobs.slurm 
│
└───reference_model
│   │   make_ball_config.py
│   
└───simulation scripts
    │   run_forward_cubic_mesh.py
    │   run_fwo_tobs.py
│
└───inputs
    |...
```
    
## Input files

--> The reference_model(s)/ folders contains either the .hdf5 arrays
    representing the reference models, or python scripts used to build the
    .hdf5 reference model file.

--> The configurations json files are either present in the input/
    folders, or python scripts that can be used to generate these files.

## simulation scripts


--> python scripts used for calculating the forward problem and running
    the inversion are in the simulation_scripts/ folder.

## shell scripts
--> shell scripts which actually call the inversion routine. One only needs to
certify that all input files are correctly targeted. 

For instance, in order to reproduce the results from Figure 7 in Section 5.1, the script to be called is ` reproducibility_integer_linear_programming/2D/section_5.1/run_single_inclusion.sh`. Its content reads:

```
#!/bin/bash

for case; do

    # get a few parameters 
    N=$(grep num_sources $case | egrep -o '[[:digit:]]{1,4}')
    rmin=$(grep '"rmin' $case | egrep -o '[[:digit:]]{0,4}\.{0,1}[[:digit:]]{0,4}')
    beta=$(grep '"beta' $case | egrep -o '[[:digit:]]{0,4}\.{0,1}[[:digit:]]{0,4}')
    gamma_m=$(grep gamma_m $case | egrep -o '[[:digit:]]{0,4}\.{0,1}[[:digit:]]{0,4}')
    gamma_v=$(grep gamma_v $case | egrep -o '[[:digit:]]{0,4}\.{0,1}[[:digit:]]{0,4}')

    # state some basic info to std
    echo "the configuration file is $case"
    echo "the number of shots is $N"
    echo "rmin = $rmin, beta = $beta, gamma_m = $gamma_m and gamma_v = $gamma_v"
    echo "hdf5 result is $resultfile, and the output directory is $outdir/"

    # run forward
    mpiexec -np $N python fwd.py -i single_inclusion.hdf5 -c $case

    # run inversion
    mpiexec -np $N python fwi_combine.py -c $case -e single_inclusion.hdf5

done
```
having the files :
*    `2D/simulation_scripts/fwd.py` 
*    `2D/simulation_scripts/fwi_combine.py`, 
*    `2D/reference_models/single_inclusion.hdf5` 
*    `2D/section_5.1/inputs/simple_2D_no_adam_no_filter/config.json`

in the same directory where the script is called, one can run 
`bash run_single_inclusion.sh config.json` 
