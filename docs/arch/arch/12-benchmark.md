---
title: Benchmark 性能日志格式
sidebar_position: 220
---

# Benchmark 性能日志格式

本节详细介绍了统一 benchmark 日志格式的架构 v0.1。该模式使得交叉引用更容易，包括与其他框架/运行、实验再现、每日性能回归数据，以及日志记录/可视化工作的分离。

## 日志格式概述

简单起见，建议优先处理 *workload*，*engine*，*hardware runtime_ms_mean* 和 *runtime_ms_std* 字段。更细粒度的日志记录，可以另外传播 **_config* 字段。

| **header** | **examples** | **category** | **notes/justification** |
|:---|:---|:---|:---|
| workload | resnet-18 | workload | name of workload |
| engine | “tvm” / “onnxruntime” | compiler |    |
| hardware | “gcp-c2-standard-16” | hardware | descriptor of target hardware environment |
| runtime_ms_mean | 12.452 | statistics |    |
| runtime_ms_std | 5.3 | statistics |    |
| timestamp | 1572282699.6 | metadata | indicates when this record is logged |
| schema_version | “0.1” | metadata | ensure reproducibility as we iterate on this schema |
| metadata | { “docker_tag”:”gcr.io/…/0a680”, … } | metadata | docker_tag is optional |
| workload_args | {“input_name”: “Input3”, “input_shape”: [list_of_shape], “data_layout”: NHCW} | workload |    |
| workload_metadata | {“class”: “vision”,”doc_url”: “https://github.com/.../README.md”, “opset”: 7,”type”: “body_analysis”,”url”: “https://onnxzoo...ferplus.tar.gz”, “md5”: “07fc7…”} | workload | source of workload |
| engine_version | “1.0.5” | compiler | use semvar format |
| engine_config | {“llvm”: “llvm-8”, “nvcc”: 10.1, “accelerator”: “MLAS”, “relay_opt_level”: 3, “tvm_target”:”llvm -mcpu=cascadelake”} | compiler | fields are optionally specified |
| compilation_config | {“opt_level”: 3, “layer_schedules”:[]/ <SHA_to_schedules>} | compiler | fields are optionally specified |
| software_config | {“os”: “ubuntu:18.04”,”pip”: { “docker”: “4.1.0”, “gitpython”: “3.0.4”, “numpy”: “1.17.4”, “onnx”: “1.6.0”}, “cudnn”: “cudnn-8”, “cuda_driver”: “480.10.1”} | backend | env dependency list |
| runtime_config | {“num_cpu_threads”: 3} | backend | info on non-hardware, non-software metadata |
| hardware_config | {“cpu_count”: 16, “cloud_machine_type”:”c2-standard-16”, “memory_GB”:64} | hardware | json descriptor of target hardware environment |
| execution_config | {“number”: 1, “repeat”: 10, “min_repeat_ms”, 0} | statistics | workload execution parameters |
| metrics | {“accuracy”: 48.5,“compilation_ms_mean”: 12} | statistics | other metrics |
| runtime_raw | [{“runtime_ms”: 12, …}, {“runtime_ms”:13,…},…] | statistics | optional raw metrics array |

## 存储格式

目前，为了可扩展性和便利性，正在将 benchmark 数据原型化为 JSON 对象，尤其是在模式的早期版本中。但是，随着我们扩大 benchmark 聚合，并稳定参数，预计会切换成列格式，例如 Arrow 或 Parquet。

以下是编码为 JSON 的示例数据：

``` json
{
  "workload":"arcface_resnet100",
  "engine":"tvm",
  "hardware":"gcp-c2-standard-16",
  "runtime_ms_mean":109.43004820081924,
  "runtime_ms_std":0.09078385126800587,
  "timestamp":"20191123003411",
  "schema_version":"0.1",
  "metadata":{
    "docker_tag":"tlcpack/ci-gpu:v0.53"
  },
  "workload_args":{
    "input_shape_dict":{
      "data":[
        1,
        3,
        112,
        112
      ]
    },
    "input_type_dict":{
      "data":"float32"
    },
    "input_value_dict":{}
  },
  "workload_metadata":{
    "class":"vision",
    "doc_url":"https://github.com/onnx/models/blob/main/vision/body_analysis/arcface/README.md",
    "md5":"66074b860f905295aab5a842be57f37d",
    "opset":8,
    "type":"body_analysis",
    "url":"https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.tar.gz"
  },
  "engine_version":"1.0.0",
  "engine_config":{},
  "compilation_config":{
    "relay_opt_level": 3
  },
  "software_config":{
    "os":"ubuntu:18.04",
    "pip":{
      "docker":"4.1.0",
      "gitpython":"3.0.4",
      "numpy":"1.17.4",
      "onnx":"1.6.0"
    }
  },
  "runtime_config":{},
  "hardware_config":{
    "cloud_machine_type":"c2-standard-16",
    "cloud_provider":"GCP",
    "cpu_count":16,
    "cpu_platform":"Intel Cascade Lake",
    "memory_GB":64
  },
  "execution_config":{},
  "metrics":{}
}
```