#!/bin/bash
    if [ -e comsol.prmtop ]
    then
      export CUDA_VISIBLE_DEVICES=0
      pmemd.cuda -O -i 01_min.in -o 01_min.out -p comsol.prmtop -c comsol.inpcrd -r 01_min.rst -ref comsol.inpcrd
    
    fi
    
    if [ -e 01_min.out ]
    then
      export CUDA_VISIBLE_DEVICES=0
      pmemd.cuda -O -i 02_min.in -o 02_min.out -p comsol.prmtop -c 01_min.rst -r 02_min.rst -ref 01_min.rst
    
    fi
    
    if [ -e 02_min.out ]
    then
      export CUDA_VISIBLE_DEVICES=0
      pmemd.cuda -O -i 03_heat.in -o 03_heat.out -p comsol.prmtop -c 02_min.rst -r 03_heat.rst -ref 02_min.rst
      
    fi
    
    if [ -e 03_heat.out ]
    then
      export CUDA_VISIBLE_DEVICES=0
      pmemd.cuda -O -i 04_density.in -o 04_density.out -p comsol.prmtop -c 03_heat.rst -r 04_density.rst -ref 03_heat.rst
      
    fi
    
    if [ -e 04_density.out ]
    then
      export CUDA_VISIBLE_DEVICES=0
      pmemd.cuda -O -i 05_equil.in -o 05_equil.out -p comsol.prmtop -c 04_density.rst -r 05_equil.rst -ref 04_density.rst
      
    fi
    
    if [ -e 05_equil.out ]
    then
      export CUDA_VISIBLE_DEVICES=0 
      pmemd.cuda -O -i 06_prod.in -o 06_prod.out -p comsol.prmtop -c 05_equil.rst -r 06_prod.rst -ref 05_equil.rst -x 06_prod.mdcrd
	  
    fi
    
	echo "DONE"
