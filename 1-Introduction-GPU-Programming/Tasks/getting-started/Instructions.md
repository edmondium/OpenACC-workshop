# JSC OpenACC Course 2024

-   Date: 29 - 31 October 2024
-   Location: *online*
-   Institue: Jülich Supercomputing Centre

## Session 1: GPU Introduction

### Task 1: Getting Started

Please open up a Terminal window via `File` → `New` → `Terminal`.

In the Terminal, make your way to `~/GPU-Course/1-Introduction-GPU-Programming/Tasks/Getting-Started`, which should already be in your account after bootstrapping the environment.

Call `source setup.sh` to load the modules of this task into your environment. They are a bit different than the modules for the other tasks!)

Run `make` to build the individual benchmark programs.

Call `./bench_all.sh` to submit the benchmarks to the batch system and
let them run on the compute nodes.

Afterwards, execute the following cells (for example with
shift+enter) to visualize the result and display the generated images. You might need to modify the content of the cell in order for Jupyter to try to
re-render the images -- just add a space at the bottom.