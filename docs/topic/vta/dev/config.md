---
title: VTA Configuration
---

The VTA stack incorporates both a hardware accelerator stack and a TVM
based software stack. VTA incorporates flexibility out of the box: by
modifying the `3rdparty/vta-hw/config/vta_config.json` high-level
configuration file, the user can change the shape of the tensor
intrinsic, clock frequency, pipelining, data type width, and on-chip
buffer sizes.

# Parameters Overview

We explain the parameters listed in the `vta_config.json` file in the
table below.

  ---------------------------------------------------------------------------
  Attribute             Format    Description
  --------------------- --------- -------------------------------------------
  `TARGET`              String    The TVM device target.

  `HW_VER`              String    VTA hardware version number.

  `LOG_INP_WIDTH`       Int       Input data type signed integer width.
                        (log2)    

  `LOG_WGT_WIDTH`       Int       Weight data type signed integer width.
                        (log2)    

  `LOG_ACC_WIDTH`       Int       Accumulator data type signed integer width.
                        (log2)    

  `LOG_BATCH`           Int       VTA matrix multiply intrinsic input/output
                        (log2)    dimension 0.

  `LOG_BLOCK`           Int       VTA matrix multiply inner dimensions.
                        (log2)    

  `LOG_UOP_BUFF_SIZE`   Int       Micro-op on-chip buffer in Bytes.
                        (log2)    

  `LOG_INP_BUFF_SIZE`   Int       Input on-chip buffer in Bytes.
                        (log2)    

  `LOG_WGT_BUFF_SIZE`   Int       Weight on-chip buffer in Bytes.
                        (log2)    

  `LOG_ACC_BUFF_SIZE`   Int       Accumulator on-chip buffer in Bytes.
                        (log2)    
  ---------------------------------------------------------------------------

> ::: note
> ::: title
> Note
> :::
>
> When a parameter name is preceded with `LOG`, it means that it
> describes a value that can only be expressed a power of two. For that
> reason we describe these parameters by their log2 value. For instance,
> to describe an integer width of 8-bits for the input data types, we
> set the `LOG_INP_WIDTH` to be 3, which is the log2 of 8. Similarly, to
> descibe a 64kB micro-op buffer, we would set `LOG_UOP_BUFF_SIZE` to be
> 16.
> :::

We provide additional detail below regarding each parameter:

> -   `TARGET`: Can be set to `"pynq"`, `"ultra96"`, `"sim"` (fast
>     simulator), or `"tsim"` (cycle accurate sim with verilator).
> -   `HW_VER`: Hardware version which increments every time the VTA
>     hardware design changes. This parameter is used to uniquely
>     identity hardware bitstreams.
> -   `LOG_BATCH`: Equivalent to A in multiplication of shape (A, B) x
>     (B, C), or typically, the batch dimension of inner tensor
>     computation.
> -   `LOG_BLOCK`: Equivalent to B and C in multiplication of shape
>     (A, B) x (B, C), or typically, the input/output channel dimensions
>     of the inner tensor computation.
