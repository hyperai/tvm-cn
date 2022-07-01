..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _docker-images:

Docker 镜像
=============
开发者可以利用 Docker 工具脚本，建立开发环境。这也有助于运行 TVM Demo 和教程。
使用 CUDA 需要用到 `Docker <https://docs.docker.com/engine/installation/>`_ 和 `nvidia-docker <https://github.com/NVIDIA/nvidia-docker/>`_。

获取 TVM 源码发行版或克隆 GitHub 仓库，以获取辅助脚本：

.. code:: bash

    git clone --recursive https://github.com/apache/tvm tvm


使用以下命令来启动 Docker 镜像：

.. code:: bash

    /path/to/tvm/docker/bash.sh <image-name>

完成本地构建后，这里的 image-name 可以是一个本地的 Docker 镜像名称，例如：`tvm.ci_cpu`。

该辅助脚本可实现：

- 挂载当前目录到 /workspace
- 将用户切换为调用 bash.sh 的用户（这样您就可以读/写主机系统）
- 在 Linux 上使用宿主机的网络。由于无法支持主机网络驱动器，请在 macOS 上使用桥接网络并暴露 8888 端口，以使用 Jupyter Notebook。


输入以下内容启动 Jupyter Notebook：

.. code:: bash

   jupyter notebook

如果你在 macOS 上启动 Jupyter Notebook 时看到报错 ``OSError: [Errno 99] Cannot assign requested address``，可通过以下方式改变绑定的 IP 地址：

.. code:: bash

   jupyter notebook --ip=0.0.0.0

注意，在 macOS 上，由于我们使用桥接网络，Jupyter Notebook 将被报告在一个类似于 ``http://{container_hostname}:8888/?token=...`` 的 URL 上运行。
在浏览器中粘贴时，需把 container_hostname 替换为 ``localhost``。


Docker 源代码
-------------
查看 `Docker 源代码 <https://github.com/apache/tvm/tree/main/docker>`_，构建自己的 Docker 镜像。

运行以下命令来构建 Docker 镜像：

.. code:: bash

    /path/to/tvm/docker/build.sh <image-name>


你也可以利用非官方的第三方预建镜像，注意：这些镜像是用来测试的，并不是 ASF 的版本。

`<https://hub.docker.com/r/tlcpack/>`_.
