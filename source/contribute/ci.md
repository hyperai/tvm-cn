---
title: Using TVM\'s CI
---

::: {.contents local=""}
:::

TVM uses Jenkins for running Linux continuous integration (CI) tests on
[branches](https://ci.tlcpack.ai/job/tvm/) and [pull
requests](https://ci.tlcpack.ai/job/tvm/view/change-requests/) through a
build configuration specified in a
[Jenkinsfile](https://github.com/apache/tvm/blob/main/Jenkinsfile).
Non-critical jobs run in GitHub Actions for Windows and MacOS jobs.

A standard CI run looks something like this viewed in [Jenkins\'
BlueOcean
viewer](https://ci.tlcpack.ai/blue/organizations/jenkins/tvm/activity).
CI runs usually take several hours to complete and pull requests (PRs)
cannot be merged before CI has successfully completed. To diagnose
failing steps, click through to the failing pipeline stage then to the
failing step to see the output logs.

![The Jenkins UI for a CI run](https://github.com/tlc-pack/web-data/raw/main/images/contribute/ci.png){width="800px"}

# Debugging Failures

When CI fails for some reason, there are several methods to diagnose the
issue.

## Jenkins Logs

The first place to look for a failure is in the CI logs, follow the red
Xs on the failing job to view the logs. Note:

-   Jenkins does not display the full log by default, at the top of the
    log viewer is a button \"Show complete log\" which will take you to
    a plaintext version of the log
-   `pytest`\_ failures are summarized at the bottom of the log but you
    will likely need to scroll up to view the actual failure.

## Reproduce Failures

Most TVM Python tests run under `pytest`\_ and can be run as described
in `pr-testing`{.interpreted-text role="ref"}.

# Keeping CI Green

Developers rely on the TVM CI to get signal on their PRs before merging.
Occasionally breakages slip through and break `main`, which in turn
causes the same error to show up on an PR that is based on the broken
commit(s). Broken commits can be identified [through
GitHub](https://github.com/apache/tvm/commits/main) via the commit
status icon or via
[Jenkins](https://ci.tlcpack.ai/blue/organizations/jenkins/tvm/activity?branch=main).
In these situations it is possible to either revert the offending commit
or submit a forward fix to address the issue. It is up to the committer
and commit author which option to choose, keeping in mind that a broken
CI affects all TVM developers and should be fixed as soon as possible.

## Skip CI for Reverts

For reverts and trivial forward fixes, adding `[skip ci]` to the
revert\'s PR title will cause CI to shortcut and only run lint.
Committers should take care that they only merge CI-skipped PRs to fix a
failure on `main` and not in cases where the submitter wants to shortcut
CI to merge a change faster. The PR title is checked when the build is
first run (specifically during the lint step, so changes after that has
run do not affect CI and will require the job to be re-triggered by
another `git push`).

``` bash
# Revert HEAD commit, make sure to insert '[skip ci]' at the beginning of
# the commit subject
git revert HEAD
git checkout -b my_fix
# After you have pushed your branch, create a PR as usual.
git push my_repo
# Example: Skip CI on a branch with an existing PR
# Adding this commit to an existing branch will cause a new CI run where
# Jenkins is skipped
git commit --allow-empty --message "[skip ci] Trigger skipped CI"
git push my_repo
```

# Handling Flaky Failures

If you notice a failure on your PR that seems unrelated to your change,
you should search [recent GitHub issues related to flaky
tests](https://github.com/apache/tvm/issues?q=is%3Aissue+%5BCI+Problem%5D+Flaky+)
and [file a new
issue](https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-problem.md&title=%5BCI+Problem%5D+)
if you don\'t see any reports of the failure. If a certain test or class
of tests affects several PRs or commits on `main` with flaky failures,
the test should be disabled via pytest\'s `@xfail` decorator\_ with
`strict=True`\_ and the relevant issue linked in the disabling PR.

``` python
@pytest.mark.xfail(strict=False, reason="Flaky test: https://github.com/apache/tvm/issues/1234")
def test_something_flaky():
    pass
```

# `ci-docker-staging`

The
[ci-docker-staging](https://github.com/apache/tvm/tree/ci-docker-staging)
branch is used to test updates to Docker images and `Jenkinsfile`
changes. When running a build for a normal PR from a forked repository,
Jenkins uses the code from the PR except for the `Jenkinsfile` itself,
which comes from the base branch. When branches are built, the
`Jenkinsfile` in the branch is used, so a committer with write access
must push PRs to a branch in apache/tvm to properly test `Jenkinsfile`
changes. If your PR makes changes to the `Jenkinsfile`, make sure to @ a
[committer](https://github.com/apache/tvm/blob/main/CONTRIBUTORS.md) and
ask them to push your PR as a branch to test the changes.

# Docker Images {#docker_images}

Each CI job runs most of its work inside a Docker container, built from
files in the [docker/](https://github.com/apache/tvm/tree/main/docker)
folder. These files are built nightly in Jenkins via the
[docker-images-ci](https://ci.tlcpack.ai/job/docker-images-ci/) job. The
images for these containers are hosted in the [tlcpack Docker
Hub](https://hub.docker.com/u/tlcpack) and referenced at the top of the
`Jenkinsfile`\_. These can be inspected and run locally via standard
Docker commands.

``` bash
# Beware: CI images can be several GB in size
# Get a bare docker shell in the ci-gpu container
docker run -it tlcpack/ci-gpu:v0.78 /bin/bash
```

`docker/bash.sh` will automatically grab the latest image from the
`Jenkinsfile` and help in mounting your current directory.

``` bash
# Run the ci_cpu image specified in Jenkinsfile
cd tvm
bash docker/bash.sh ci_cpu
# the tvm directory is automatically mounted
# example: build tvm (note: this will overrwrite build/)
$ ./tests/scripts/task_config_build_cpu.sh
$ ./tests/scripts/task_build.sh build -j32
```

# Reporting Issues

Issues with CI should be [reported on
GitHub](https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-problem.md&title=%5BCI+Problem%5D+)
with a link to the relevant jobs, commits, or PRs.
