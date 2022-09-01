variable "DEFAULT_TAG" {
  default = ["openbayes/tvm-cn:local"]
}

# Special target: https://github.com/docker/metadata-action#bake-definition
target "docker-metadata-action" {
  tags = "${DEFAULT_TAG}"
}

# Default target if none specified
group "default" {
  targets = ["build-local"]
}

target "build" {
  inherits = ["docker-metadata-action"]
}

target "build-local" {
  inherits = ["build"]
  output = ["type=docker"]
}

target "build-all" {
  inherits = ["build"]
  platforms = [
    "linux/amd64",
  ]
}
