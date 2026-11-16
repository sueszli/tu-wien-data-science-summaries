# Compare GIL and native-code benchmark results


An Efficient Programs project (185.190, winter 2024) on Python 3.13, threading and native code. The [original public project](https://github.com/sueszli/nogil) compares sequential Python, Python threads and processes, C via ctypes, and a CPython extension on a small hash-search workload.

[Selected counters](results.csv) are medians transcribed from the original report. The manually managed threaded implementation retired roughly 5.2 times fewer instructions with the GIL disabled. The executor-based threaded implementation did not show the same pattern. The CPython/OpenMP combination retired the fewest instructions of the configurations shown here, but this comparison also changes the implementation, so that result is not an isolated GIL effect.

The benchmark ran on an Intel Core i7-1165G7. Instructions are not elapsed time, and these figures are not general Python performance claims. The full harness, source and report remain in the public standalone project.
