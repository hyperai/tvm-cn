---
title: microTVM：裸机上的 TVM
---

microTVM runs TVM models on bare-metal (i.e. IoT) devices. microTVM
depends only on the C standard library, and doesn\'t require an
operating system to execute. microTVM is currently under heavy
development.

![](https://raw.githubusercontent.com/tvmai/web-data/main/images/dev/microtvm_workflow.svg){.align-center
width="85.0%"}

microTVM is:

-   an extension to TVM\'s compiler to allow it to target
    microcontrollers
-   a way to run the TVM RPC server on-device, to allow autotuning
-   a minimal C runtime that supports standalone model inference on bare
    metal devices.

## Supported Hardware

microTVM currently tests against Cortex-M microcontrollers with the
Zephyr RTOS; however, it is flexible and portable to other processors
such as RISC-V and does not require Zephyr. The current demos run
against QEMU and the following hardware:

-   [STM
    Nucleo-F746ZG](https://www.st.com/en/evaluation-tools/nucleo-f746zg.html)
-   [STM STM32F746
    Discovery](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)
-   [nRF 5340 Development
    Kit](https://www.nordicsemi.com/Software-and-tools/Development-Kits/nRF5340-DK)

## Getting Started with microTVM

Before working with microTVM, we recommend you have a supported
development board. Then, follow these tutorials to get started with
microTVM:

1.  `Start the microTVM Reference VM <tutorial-micro-reference-vm>`{.interpreted-text
    role="ref"}. The microTVM tutorials depend on Zephyr and on a
    compiler toolchain for your hardware. The reference VM is a
    convenient way to install those dependencies.
2.  Try the
    `microTVM with TFLite Tutorial <microTVM-with-TFLite>`{.interpreted-text
    role="ref"}.
3.  Try running a more complex [CIFAR10-CNN
    model](https://github.com/areusch/microtvm-blogpost-eval).

## How microTVM Works

You can read more about the design of these pieces at the
`microTVM Design Document <microTVM-design>`{.interpreted-text
role="ref"}.

## Help and Discussion

The [TVM Discuss Forum](https://discuss.tvm.ai) is a great place to
collaborate on microTVM tasks, and maintains a searchable history of
past problems.
