# HLS 后端示例

TVM 支持带有 SDAccel 的 Xilinx FPGA 板，接下来介绍如何将 TVM 部署到 AWS F1 FPGA 实例。

:::note
此功能仍处于测试阶段，目前无法用 SDAccel 部署端到端神经网络。
:::

本教程使用了两个 Python 脚本：

* build.py - 用于合成 FPGA 比特流的脚本。

   ``` python
     import tvm
     from tvm import te
     
     tgt= tvm.target.Target("sdaccel", host="llvm")
     
     n = te.var("n")
     A = te.placeholder((n,), name='A')
     B = te.placeholder((n,), name='B')
     C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
     
     s = te.create_schedule(C.op)
     px, x = s[C].split(C.op.axis[0], nparts=1)
     
     s[C].bind(px, tvm.te.thread_axis("pipeline"))
     
     fadd = tvm.build(s, [A, B, C], tgt, name="myadd")
     fadd.save("myadd.o")
     fadd.imported_modules[0].save("myadd.xclbin")
     
     tvm.contrib.cc.create_shared("myadd.so", ["myadd.o"])
   ```

* run.py - 将 FPGA 作为加速器的脚本。

   ``` python
     import tvm
     import numpy as np
     import os
     
     tgt = "sdaccel"
     
     fadd = tvm.runtime.load_module("myadd.so")
     if os.environ.get("XCL_EMULATION_MODE"):
         fadd_dev = tvm.runtime.load_module("myadd.xclbin")
     else:
         fadd_dev = tvm.runtime.load_module("myadd.awsxclbin")
     fadd.import_module(fadd_dev)
     
     dev = tvm.device(tgt, 0)
     
     n = 1024
     a = tvm.nd.array(np.random.uniform(size=n).astype("float32"), dev)
     b = tvm.nd.array(np.random.uniform(size=n).astype("float32"), dev)
     c = tvm.nd.array(np.zeros(n, dtype="float32"), dev)
     
     fadd(a, b, c)
     tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
   ```

## 设置

* 用 FPGA Developer AMI 启动实例。无需 F1 实例来进行仿真和合成，因此推荐用开销较低的实例。
* 设置 AWS FPGA 开发套件：

   ```  bash
     git clone https://github.com/aws/aws-fpga.git
     cd aws-fpga
     source sdaccel_setup.sh
     source ${XILINX_SDX}/settings64.sh
   ```

* 启用 OpenCL 前设置 TVM。

## 仿真

* 为仿真创建 emconfig.json：

   ``` bash
     emconfigutil --platform ${AWS_PLATFORM} --nd 1
   ```

* 将 emconfig.json 复制到 Python binary 目录下：因为当前的 Xilinx 工具包假定宿主机的二进制文件和 emconfig.json 文件处于同一路径。

   ``` bash
     cp emconfig.json $(dirname $(which python))
   ```

* 运行软件仿真：

   ``` bash
     export XCL_EMULATION_MODE=1
     export XCL_TARGET=sw_emu
     
     python build.py
     python run.py
   ```

* 运行硬件仿真：

   ``` bash
     export XCL_EMULATION_MODE=1
     export XCL_TARGET=hw_emu
     
     python build.py
     python run.py
   ```

## 合成

* 用以下脚本进行合成：

   ``` bash
     unset XCL_EMULATION_MODE
     export XCL_TARGET=hw
     
     python build.py
   ```

* 创建 AWS FPGA 镜像，并将其上传到 AWS S3：

   ``` bash
     ${SDACCEL_DIR}/tools/create_sdaccel_afi.sh \
         -xclbin=myadd.xclbin -o=myadd \
         -s3_bucket=<bucket-name> -s3_dcp_key=<dcp-folder-name> \
         -s3_logs_key=<logs-folder-name>
   ```

这会生成 awsxclbin 文件（在 F1 实例上使用 AWS FPGA 镜像必需）。

## 运行

* 启动 Amazon EC2 F1 实例。
* 将 `myadd.so`，`myadd.awsxclbin` 和 `run.py` 复制到 F1 实例中。
* 设置 AWS FPGA 开发套件：

   ``` bash
     git clone https://github.com/aws/aws-fpga.git
     cd aws-fpga
     source sdaccel_setup.sh
   ```

* 启用 OpenCL 前设置 TVM。
* 以 root 身份设置环境变量：

   ``` bash
     sudo sh
     source ${INSTALL_ROOT}/setup.sh
   ```

* 运行：

   ``` bash
     python run.py
   ```