# Modules needed for JSC GPU COURSE
# Please source this file like "source setup.sh"
module purge
module load Intel/2023.2.1
module load ParaStationMPI/5.9.2-1
module load CUDA/12
#module load imkl
module load Python/3.11.3
module load SciPy-bundle/2023.07
module load matplotlib/3.7.2
module load OpenGL/2023a
module load freeglut/.3.4.0

export JSCCOURSE_TASK1_MODULES_SOURCED=1
