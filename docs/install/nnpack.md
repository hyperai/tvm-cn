---
title: NNPACK Contrib Installation
---

[NNPACK](https://github.com/Maratyszcza/NNPACK) is an acceleration
package for neural network computations, which can run on x86-64, ARMv7,
or ARM64 architecture CPUs. Using NNPACK, higher-level libraries like
\_\_MXNet\_\_ can speed up the execution on multi-core CPU computers,
including laptops and mobile devices.

::: note
::: title
Note
:::

AS TVM already has natively tuned schedules, NNPACK is here mainly for
reference and comparison purpose. For regular use prefer native tuned
TVM implementation.
:::

TVM supports NNPACK for forward propagation (inference only) in
convolution, max-pooling, and fully-connected layers. In this document,
we give a high level overview of how to use NNPACK with TVM.

# Conditions

The underlying implementation of NNPACK utilizes several acceleration
methods, including fft and winograd. These algorithms work better on
some special [batch size]{.title-ref}, [kernel size]{.title-ref}, and
[stride]{.title-ref} settings than on other, so depending on the
context, not all convolution, max-pooling, or fully-connected layers can
be powered by NNPACK. When favorable conditions for running NNPACKS are
not met,

NNPACK only supports Linux and OS X systems. Windows is not supported at
present.

# Build/Install NNPACK

If the trained model meets some conditions of using NNPACK, you can
build TVM with NNPACK support. Follow these simple steps:

uild NNPACK shared library with the following commands. TVM will link
NNPACK dynamically.

Note: The following NNPACK installation instructions have been tested on
Ubuntu 16.04.

## Build Ninja

NNPACK need a recent version of Ninja. So we need to install ninja from
source.

``` bash
git clone git://github.com/ninja-build/ninja.git
cd ninja
./configure.py --bootstrap
```

Set the environment variable PATH to tell bash where to find the ninja
executable. For example, assume we cloned ninja on the home directory
\~. then we can added the following line in \~/.bashrc.

``` bash
export PATH="${PATH}:~/ninja"
```

## Build NNPACK

The new CMAKE version of NNPACK download
[Peach](https://github.com/Maratyszcza/PeachPy) and other dependencies
alone

Note: at least on OS X, running [ninja install]{.title-ref} below will
overwrite googletest libraries installed in
[/usr/local/lib]{.title-ref}. If you build googletest again to replace
the nnpack copy, be sure to pass [-DBUILD_SHARED_LIBS=ON]{.title-ref} to
[cmake]{.title-ref}.

``` bash
git clone --recursive https://github.com/Maratyszcza/NNPACK.git
cd NNPACK
# Add PIC option in CFLAG and CXXFLAG to build NNPACK shared library
sed -i "s|gnu99|gnu99 -fPIC|g" CMakeLists.txt
sed -i "s|gnu++11|gnu++11 -fPIC|g" CMakeLists.txt
mkdir build
cd build
# Generate ninja build rule and add shared library in configuration
cmake -G Ninja -D BUILD_SHARED_LIBS=ON ..
ninja
sudo ninja install

# Add NNPACK lib folder in your ldconfig
echo "/usr/local/lib" > /etc/ld.so.conf.d/nnpack.conf
sudo ldconfig
```

# Build TVM with NNPACK support

``` bash
git clone --recursive https://github.com/apache/tvm tvm
```

-   Set [set(USE_NNPACK ON)]{.title-ref} in config.cmake.
-   Set [NNPACK_PATH]{.title-ref} to the \$(YOUR_NNPACK_INSTALL_PATH)

after configuration use [make]{.title-ref} to build TVM

``` bash
make
```
