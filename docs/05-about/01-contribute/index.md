---

title: 贡献者指南

---


TVM 由社区成员共同开发，欢迎任何人参与贡献。我们重视各类贡献形式，包括但不限于：
* 对现有补丁进行代码审查。
*  文档编写和使用示例。
*  在论坛和 issue 中的社区参。
* 提高代码可读性和撰写开发者指南。
   *  欢迎添加注释以提高代码可读性。
   *  欢迎撰写文档，解释内部设计决策。
*  编写测试用例以增强代码库的健壮性。
*  编写教程、博客文章、演讲以推广项目。


以下是各方面贡献的指导：
* [TVM 社区准则](https://tvm.hyper.ai/docs/about/contribute/TVM-Community-community)
   * [通用开发流程](https://tvm.apache.org/docs/contribute/community.html#general-development-process)
   * [策略决策流程](https://tvm.apache.org/docs/contribute/community.html#strategy-decision-process)
   * [提交者 (Committers)](https://tvm.apache.org/docs/contribute/community.html#committers)
   * [审阅者 (Reviewers)](https://tvm.apache.org/docs/contribute/community.html#reviewers)
* [提交 Pull Request](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request)
   * [指南](https://tvm.apache.org/docs/contribute/pull_request.html#guidelines)
   * [提交信息指南](https://tvm.apache.org/docs/contribute/pull_request.html#commit-message-guideline)
   * [CI 环境](https://tvm.apache.org/docs/contribute/pull_request.html#ci-environment)
   * [测试](https://tvm.apache.org/docs/contribute/pull_request.html#testing)
* [代码审查](https://tvm.hyper.ai/docs/about/contribute/code_review)
   * [建立信任](https://tvm.apache.org/docs/contribute/code_review.html#building-trust)
   * [社区参与](https://tvm.apache.org/docs/contribute/code_review.html#community-participation)
   * [仔细阅读代码](https://tvm.apache.org/docs/contribute/code_review.html#read-the-code-carefully)
   * [保持尊重](https://tvm.apache.org/docs/contribute/code_review.html#be-respectful)
   * [代码质量考量因素](https://tvm.apache.org/docs/contribute/code_review.html#factors-to-consider-about-code-quality)
   * [达成共识](https://tvm.apache.org/docs/contribute/code_review.html#consensus-building)
   * [一致性](https://tvm.apache.org/docs/contribute/code_review.html#consistency)
   * [其他建议](https://tvm.apache.org/docs/contribute/code_review.html#additional-recommendations)
* [提交者指南](https://tvm.hyper.ai/docs/about/contribute/committer_guide)
   * [社区优先](https://tvm.apache.org/docs/contribute/committer_guide.html#community-first)
   * [公开归档原则](https://tvm.apache.org/docs/contribute/committer_guide.html#public-archive-principle)
   * [独立项目管理](https://tvm.apache.org/docs/contribute/committer_guide.html#independent-project-management)
   * [指导 Pull Request](https://tvm.apache.org/docs/contribute/committer_guide.html#shepherd-a-pull-request)
   * [时间管理](https://tvm.apache.org/docs/contribute/committer_guide.html#time-management)
   * [广泛协作](https://tvm.apache.org/docs/contribute/committer_guide.html#broad-collaboration)
* [文档撰写](https://tvm.hyper.ai/docs/about/contribute/documentation)
   * [四种文档类型](https://tvm.apache.org/docs/contribute/document.html#the-four-document-types)
   * [技术细节](https://tvm.apache.org/docs/contribute/document.html#technical-details)
* [代码规范与技巧](https://tvm.hyper.ai/docs/about/contribute/code_guide_and_Tips)
   * [C++ 代码风格](https://tvm.apache.org/docs/contribute/code_guide.html#c-code-styles)
   * [Python 代码风格](https://tvm.apache.org/docs/contribute/code_guide.html#python-code-styles)
   * [编写 Python 测试](https://tvm.apache.org/docs/contribute/code_guide.html#writing-python-tests)
   * [网络资源](https://tvm.apache.org/docs/contribute/code_guide.html#network-resources)
   * [整数常量表达式处理](https://tvm.apache.org/docs/contribute/code_guide.html#handle-integer-constant-expression)
* [Git 使用技巧](https://tvm.hyper.ai/docs/about/contribute/Git_Usage_Tips)
   * [如何解决与 main 分支的冲突](https://tvm.apache.org/docs/contribute/git_howto.html#how-to-resolve-a-conflict-with-main)
   * [如何合并多个提交](https://tvm.apache.org/docs/contribute/git_howto.html#how-to-combine-multiple-commits-into-one)
   * [重置到最新的 main 分支](https://tvm.apache.org/docs/contribute/git_howto.html#reset-to-the-most-recent-main-branch)
   * [恢复重置前的提交](https://tvm.apache.org/docs/contribute/git_howto.html#recover-a-previous-commit-after-reset)
   * [仅将最新的 k 个提交到 main 分支](https://tvm.apache.org/docs/contribute/git_howto.html#apply-only-k-latest-commits-on-to-the-main)
   * [强制推送的后果](https://tvm.apache.org/docs/contribute/git_howto.html#what-is-the-consequence-of-force-push)
* [使用 TVM 的 CI](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci)
   * [贡献者指南](https://tvm.apache.org/docs/contribute/ci.html#for-contributors)
   * [维护者指南](https://tvm.apache.org/docs/contribute/ci.html#for-maintainers)
* [发布流程](https://tvm.hyper.ai/docs/about/contribute/Release_Process)
   * [准备发布说明](https://tvm.apache.org/docs/contribute/release_process.html#prepare-the-release-notes)
   * [准备候选版本](https://tvm.apache.org/docs/contribute/release_process.html#prepare-the-release-candidate)
   * [准备 GPG 密钥](https://tvm.apache.org/docs/contribute/release_process.html#prepare-the-gpg-key)
   * [创建候选版本](https://tvm.apache.org/docs/contribute/release_process.html#cut-a-release-candidate)
   * [在 main 上更新 TVM 版本](https://tvm.apache.org/docs/contribute/release_process.html#update-tvm-version-on-main)
   * [上传发布候选版](https://tvm.apache.org/docs/contribute/release_process.html#upload-the-release-candidate)
   * [拣选提交](https://tvm.apache.org/docs/contribute/release_process.html#cherry-picking)
   * [发起候选版本投票](https://tvm.apache.org/docs/contribute/release_process.html#call-a-vote-on-the-release-candidate)
   * [完成发布](https://tvm.apache.org/docs/contribute/release_process.html#post-the-release)
   * [更新 TVM 网站](https://tvm.apache.org/docs/contribute/release_process.html#update-the-tvm-website)
   * [发布公告](https://tvm.apache.org/docs/contribute/release_process.html#post-the-announcement)
   * [补丁发布](https://tvm.apache.org/docs/contribute/release_process.html#patch-releases)
* [错误处理指南](https://tvm.hyper.ai/docs/about/contribute/error_handling-guide)
   * [在 C++ 中抛出特定错误](https://tvm.apache.org/docs/contribute/error_handling.html#raise-a-specific-error-in-c)
   * [如何选择错误类型](https://tvm.apache.org/docs/contribute/error_handling.html#how-to-choose-an-error-type)


