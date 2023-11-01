---
title: microTVM 虚拟机参考手册
---

# microTVM 虚拟机参考手册

:::note
单击 [此处](https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_reference_vm.html#sphx-glr-download-how-to-work-with-microtvm-micro-reference-vm-py) 下载完整的示例代码
:::

**作者**：[Andrew Reusch](mailto:areusch%40octoml.ai)

本教程介绍如何参考虚拟机启动 microTVM，可以使用虚拟机在真实的物理硬件上进行开发，而无需单独安装 microTVM 依赖项，这种方法在使用 microTVM 重现行为时（如提交错误报告）也特别有用。

microTVM 使得 TVM 可以在裸机微控制器上构建和执行模型。 它旨在与各种 SoC 和 runtime 环境（即裸机、RTOS 等）兼容，microTVM 虚拟机参考手册可提供稳定环境来允许开发者共享和重现错误及结果。

## 工作原理

虚拟机没有存储到 TVM 仓库中——然而，`apps/microtvm/reference-vm` 中的文件描述了如何将虚拟机构建到 [Vagrant](https://vagrantup.com/) 虚拟机构建器工具。

VMs 参考手册分为两个部分：

1. Vagrant Base Box 包含该平台的所有稳定依赖，构建脚本存储在 `apps/microtvm/reference-vm/<platform>/base-box` 中，当平台的“稳定”依赖发生变化时，TVM committers 会运行这些，并且生成的基本 boxes 存储在 [Vagrant Cloud](https://app.vagrantup.com/tlcpack) 中。
2. 通常用 Base Box 作为起点构建每个工作空间的虚拟机。构建脚本存储在 `apps/microtvm/reference-vm/<platform>` （除了 `base-box` 之外的所有内容）。

## 设置虚拟机

### 安装必要软件

确保最少安装以下软件：

1. [Vagrant](https://vagrantup.com/)
2. 虚拟机监控器（**VirtualBox**、**Parallels** 或 **VMWare Fusion/Workstation**）。推荐使用 [VirtualBox](https://www.virtualbox.org/)，它是一个免费的虚拟机监控器，注意，USB 转发需要安装 [VirtualBox 扩展包](https://www.virtualbox.org/wiki/Downloads#VirtualBox6.1.16OracleVMVirtualBoxExtensionPack) 。若使用 VirtualBox，还要安装 [vbguest](https://github.com/dotless-de/vagrant-vbguest) 插件。
3. 若虚拟机监控器需要，可下载 [Vagrant 提供的插件](https://github.com/hashicorp/vagrant/wiki/Available-Vagrant-Plugins#providers)（或查看 [这里](https://www.vagrantup.com/vmware) 获取 VMWare 相关信息）。

### 首次启动

首次使用虚拟机参考手册时，要在本地创建 box，并对其进行配置。

``` bash
# 如果不使用 Zephyr，将 zephyr 替换为不同平台的名称。
~/.../tvm $ cd apps/microtvm/reference-vm/zephyr
# 将 <provider_name> 替换为要用的管理程序的名称（即 virtualbox、parallels、vmware_desktop）。
~/.../tvm/apps/microtvm/reference-vm/zephyr $ vagrant up --provider=<provider_name>
```

此命令需要几分钟运行，并且需要4到5GB的存储空间，它执行的内容如下：

1. 下载 [microTVM base box](https://app.vagrantup.com/tlcpack/boxes/microtvm) 并克隆它，形成特定于该 TVM 目录的新虚拟机。
2. 把 TVM 目录（若使用 `git-subtree`，原始 `.git` 仓库）挂载到虚拟机中。
3. 构建 TVM 并安装一个 Python virtualenv，其包含的依赖与 TVM 构建相对应。

### 将硬件连接到虚拟机

接下来配置 USB，将物理开发单板连接到虚拟机（而非直接连接到笔记本电脑的操作系统）。

推荐设置一个设备过滤器，而非一次性转发，因为编程时设备可能会重启，此时需要再次启用转发。这样做的好处是最终用户不会明显有感觉。参考教程：

* [VirtualBox](https://www.virtualbox.org/manual/ch03.html#usb-support)
* [Parallels](https://kb.parallels.com/122993)
* [VMWare Workstation](https://docs.vmware.com/en/VMware-Workstation-Pro/15.0/com.vmware.ws.using.doc/GUID-E003456F-EB94-4B53-9082-293D9617CB5A.html)

### 在虚拟机参考手册中重建 TVM

首次启动后，确保在修改 C++ runtime 或 checkout 不同版本时，在 `$TVM_HOME/build-microtvm-zephyr` 中保持构建是最新的。可以重新配置机器（在运行 `vagrant up` 之前在同一目录中运行 `vagrant provision`）或自己手动重建 TVM。

注意：在虚拟机中构建的 TVM `.so` 可能与在主机上使用的不同，这就是它被构建在特殊目录 `build-microtvm-zephyr` 中的原因。

### 登录虚拟机

虚拟机应该仅对主机名为 `microtvm` 的主机可用，通过 SSH 连接到虚拟机：

``` bash
$ vagrant ssh
```

然后 `cd` 到主机上用于 TVM 的相同路径，例如，在 Mac 上：

``` bash
$ cd /Users/yourusername/path/to/tvm
```

## 运行测试

配置虚拟机后，可用 `poetry` 执行测试：

``` bash
$ cd apps/microtvm/reference-vm/zephyr
$ poetry run python3 ../../../../tests/micro/zephyr/test_zephyr.py --zephyr-board=stm32f746g_disco
```

若没有连接物理硬件，但要用本地 QEMU 模拟器（在虚拟机中运行）运行测试，使用以下命令：

``` bash
$ cd /Users/yourusername/path/to/tvm
$ cd apps/microtvm/reference-vm/zephyr/
$ poetry run pytest ../../../../tests/micro/zephyr/test_zephyr.py --zephyr-board=qemu_x86
```

[下载 Python 源代码：micro_reference_vm.py](https://tvm.apache.org/docs/_downloads/79027b28c061178b7ea56e3f047eeef1/micro_reference_vm.py)

[下载 Jupyter notebook：micro_reference_vm.ipynb](https://tvm.apache.org/docs/_downloads/7ef06253b3d2676eb50e20a5f81ef8f9/micro_reference_vm.ipynb)