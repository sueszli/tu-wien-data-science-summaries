# Render a Julia set with a parallel pixel loop


A small companion to Basics of Parallel Computing (191.114). The original project explored OpenMP and CUDA for Julia-set rendering. Each pixel can be computed independently, so the outer two loops can be parallelized without a shared reduction.

The example writes a portable grayscale image. The serial build needs a C99 compiler. On a compiler with OpenMP support, add the OpenMP flag and compare output to the serial build.

    cc -std=c99 -O2 julia.c -o julia
    ./julia > julia.pgm

The example does not include the original CUDA code, assessment submission, or cluster output.
