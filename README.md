# Automatic Differentiation Library

This repository provides a simple **Automatic Differentiation (AD)** library in C++ for building computational graphs and performing gradient-based computations.

## Features

- **Core Classes**:
  - `Var`: Represents variables in the computational graph.
  - `VarData`: Stores variable values, gradients, and dependencies.
- **Supported Operations**:
  - Arithmetic: `+`, `-`, `*`, `/` (supports `Var` and `double`).
  - Mathematical functions: `exp`, `log`, `sqrt`.
- **Gradient Computation**:
  - Backpropagation using `Var::backward()`.

## File Overview

- `ad.hh`: Header file defining the `Var` and `VarData` classes.
- `ad.cc`: Implementation of the library.
- `Makefile`: Build system for compiling the library and tests.
- `test_var.cc`: Example tests for the library (not shown here).

## Build and Run

1. Compile the library and tests:
   ```bash
   make
   ```
2. Run the compiled binary:
   ```bash
   ./ad
   ```
3. Clean up:
   ```bash
   make clean
   ```

## Example Usage

```cpp
#include "ad.hh"
#include <iostream>

int main() {
    Var x(2.0); // Create a variable x = 2.0
    Var y = x * x + 3.0; // y = x^2 + 3
    y.backward(); // Compute gradients

    std::cout << "Value of y: " << y.value() << std::endl; // Output: 7
    std::cout << "Gradient of x: " << x.grad() << std::endl; // Output: 4
    return 0;
}
```

## License

This project is licensed under the MIT License.
```