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
   * [通用开发流程](https://tvm.hyper.ai/docs/about/contribute/TVM-Community-community#%E9%80%9A%E7%94%A8%E5%BC%80%E5%8F%91%E6%B5%81%E7%A8%8B)
   * [战略决策流程](https://tvm.hyper.ai/docs/about/contribute/TVM-Community-community#%E6%88%98%E7%95%A5%E5%86%B3%E7%AD%96%E6%B5%81%E7%A8%8B) 
   * [提交者](https://tvm.hyper.ai/docs/about/contribute/TVM-Community-community#%E6%8F%90%E4%BA%A4%E8%80%85) 
   * [审阅者](https://tvm.hyper.ai/docs/about/contribute/TVM-Community-community#%E6%8F%90%E4%BA%A4%E8%80%85)
* [提交 Pull Request](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request)
   * [指南](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request#%E6%8C%87%E5%8D%97)
   * [提交信息指南](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request#%E6%8F%90%E4%BA%A4%E4%BF%A1%E6%81%AF%E6%8C%87%E5%8D%97)
   * [持续集成（CI）环境](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request#ci-%E7%8E%AF%E5%A2%83)
   * [测试](https://tvm.hyper.ai/docs/about/contribute/Submit_a_pull_request#ci-%E7%8E%AF%E5%A2%83)
* [代码审查](https://tvm.hyper.ai/docs/about/contribute/code_review)
   * [建立信任](https://tvm.hyper.ai/docs/about/contribute/code_review#%E5%BB%BA%E7%AB%8B%E4%BF%A1%E4%BB%BB)
   * [社区参与](https://tvm.hyper.ai/docs/about/contribute/code_review#%E7%A4%BE%E5%8C%BA%E5%8F%82%E4%B8%8Ecommunity-participation)
   * [仔细阅读代码](https://tvm.hyper.ai/docs/about/contribute/code_review#%E8%AE%A4%E7%9C%9F%E9%98%85%E8%AF%BB%E4%BB%A3%E7%A0%81)
   * [保持尊重](https://tvm.hyper.ai/docs/about/contribute/code_review#%E4%BF%9D%E6%8C%81%E5%B0%8A%E9%87%8D)
   * [代码质量考量因素](https://tvm.hyper.ai/docs/about/contribute/code_review#%E4%BB%A3%E7%A0%81%E8%B4%A8%E9%87%8F%E8%80%83%E9%87%8F%E5%9B%A0%E7%B4%A0)
   * [达成共识](https://tvm.hyper.ai/docs/about/contribute/code_review#%E4%BB%A3%E7%A0%81%E8%B4%A8%E9%87%8F%E8%80%83%E9%87%8F%E5%9B%A0%E7%B4%A0)
   * [一致性](https://tvm.hyper.ai/docs/about/contribute/code_review#%E4%B8%80%E8%87%B4%E6%80%A7)
   * [其他建议](https://tvm.hyper.ai/docs/about/contribute/code_review#%E5%85%B6%E4%BB%96%E5%BB%BA%E8%AE%AE)
* [提交者指南](https://tvm.hyper.ai/docs/about/contribute/committer_guide)
   * [社区优先](https://tvm.hyper.ai/docs/about/contribute/committer_guide#%E7%A4%BE%E5%8C%BA%E4%BC%98%E5%85%88)
   * [公开归档原则](https://tvm.hyper.ai/docs/about/contribute/committer_guide#%E5%85%AC%E5%BC%80%E5%BD%92%E6%A1%A3%E5%8E%9F%E5%88%99)
   * [独立项目管理](https://tvm.hyper.ai/docs/about/contribute/committer_guide#%E7%8B%AC%E7%AB%8B%E9%A1%B9%E7%9B%AE%E7%AE%A1%E7%90%86)
   * [指导 Pull Request（PR）](https://tvm.hyper.ai/docs/about/contribute/committer_guide#%E6%8C%87%E5%AF%BC-pull-requestpr)
   * [时间管理](https://tvm.hyper.ai/docs/about/contribute/committer_guide#%E6%97%B6%E9%97%B4%E7%AE%A1%E7%90%86)
   * [广泛协作](https://tvm.hyper.ai/docs/about/contribute/committer_guide#%E5%B9%BF%E6%B3%9B%E5%8D%8F%E4%BD%9C)
* [文档撰写](https://tvm.hyper.ai/docs/about/contribute/documentation)
   * [四种文档类型](https://tvm.hyper.ai/docs/about/contribute/documentation#%E5%9B%9B%E7%A7%8D%E6%96%87%E6%A1%A3%E7%B1%BB%E5%9E%8B)
   * [技术细节](https://tvm.hyper.ai/docs/about/contribute/documentation#%E6%8A%80%E6%9C%AF%E7%BB%86%E8%8A%82)
* [代码规范与技巧](https://tvm.hyper.ai/docs/about/contribute/code_guide_and_Tips)
   * [C++ 代码风格](https://tvm.hyper.ai/docs/about/contribute/code_guide_and_Tips#c-%E4%BB%A3%E7%A0%81%E9%A3%8E%E6%A0%BC)
   * [Python 代码风格](https://tvm.hyper.ai/docs/about/contribute/code_guide_and_Tips#python-%E4%BB%A3%E7%A0%81%E6%A0%B7%E5%BC%8F)
   * [编写 Python 测试](https://tvm.hyper.ai/docs/about/contribute/code_guide_and_Tips#%E7%BC%96%E5%86%99-python-%E6%B5%8B%E8%AF%95)
   * [网络资源](https://tvm.hyper.ai/docs/about/contribute/code_guide_and_Tips#%E7%BD%91%E7%BB%9C%E8%B5%84%E6%BA%90%E5%A4%84%E7%90%86)
   * [整型常量表达式处理](https://tvm.hyper.ai/docs/about/contribute/code_guide_and_Tips#%E6%95%B4%E5%9E%8B%E5%B8%B8%E9%87%8F%E8%A1%A8%E8%BE%BE%E5%BC%8F%E5%A4%84%E7%90%86)
* [Git 使用技巧](https://tvm.hyper.ai/docs/about/contribute/Git_Usage_Tips)
   * [如何解决与 main 分支冲突](https://tvm.hyper.ai/docs/about/contribute/Git_Usage_Tips#%E5%A6%82%E4%BD%95%E8%A7%A3%E5%86%B3%E4%B8%8Emain%E5%88%86%E6%94%AF%E7%9A%84%E5%86%B2%E7%AA%81) 
   * [如何合并多个提交](https://tvm.hyper.ai/docs/about/contribute/Git_Usage_Tips#%E5%A6%82%E4%BD%95%E5%90%88%E5%B9%B6%E5%A4%9A%E4%B8%AA%E6%8F%90%E4%BA%A4)
   * [重置到最新的 main 分支](https://tvm.hyper.ai/docs/about/contribute/Git_Usage_Tips#%E9%87%8D%E7%BD%AE%E5%88%B0%E6%9C%80%E6%96%B0%E7%9A%84-main-%E5%88%86%E6%94%AF)
   * [恢复重置前的提交](https://tvm.hyper.ai/docs/about/contribute/Git_Usage_Tips#%E9%87%8D%E7%BD%AE%E5%88%B0%E6%9C%80%E6%96%B0%E7%9A%84-main-%E5%88%86%E6%94%AF)
   * [仅将最新的 k 个提交到 main 分支](https://tvm.hyper.ai/docs/about/contribute/Git_Usage_Tips#%E4%BB%85%E5%B0%86%E6%9C%80%E6%96%B0%E7%9A%84-k-%E4%B8%AA%E6%8F%90%E4%BA%A4%E5%88%B0-main-%E5%88%86%E6%94%AF)
   * [强制推送的后果](https://tvm.hyper.ai/docs/about/contribute/Git_Usage_Tips#%E5%BC%BA%E5%88%B6%E6%8E%A8%E9%80%81%E7%9A%84%E5%90%8E%E6%9E%9C)
* [使用 TVM 的 CI](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci)
   * [贡献者指南](https://tvm.hyper.ai/docs/about/contribute/Git_Usage_Tips#%E5%BC%BA%E5%88%B6%E6%8E%A8%E9%80%81%E7%9A%84%E5%90%8E%E6%9E%9C)
   * [维护者指南](https://tvm.hyper.ai/docs/about/contribute/Using_TVM's_Ci#%E7%BB%B4%E6%8A%A4%E8%80%85%E6%8C%87%E5%8D%97)
* [发布流程](https://tvm.hyper.ai/docs/about/contribute/Release_Process)
   * [准备发布说明](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E5%87%86%E5%A4%87%E5%8F%91%E5%B8%83%E8%AF%B4%E6%98%8E)
   * [准备候选版本](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E5%87%86%E5%A4%87%E5%80%99%E9%80%89%E7%89%88%E6%9C%AC)
   * [准备 GPG 密钥](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E5%87%86%E5%A4%87-gpg-%E5%AF%86%E9%92%A5)
   * [创建候选版本](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E5%88%9B%E5%BB%BA%E5%80%99%E9%80%89%E7%89%88%E6%9C%AC%E5%88%86%E6%94%AF)
   * [在 main 上更新 TVM 版本](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E5%9C%A8-main-%E4%B8%8A%E6%9B%B4%E6%96%B0-tvm-%E7%89%88%E6%9C%AC)
   * [上传候选版本](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E4%B8%8A%E4%BC%A0%E5%80%99%E9%80%89%E7%89%88%E6%9C%AC)
   * [拣选提交](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E6%8B%A3%E9%80%89%E6%8F%90%E4%BA%A4)
   * [发起候选版本投票](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E5%8F%91%E8%B5%B7%E5%80%99%E9%80%89%E7%89%88%E6%9C%AC%E6%8A%95%E7%A5%A8)
   * [完成发布](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E5%AE%8C%E6%88%90%E5%8F%91%E5%B8%83)
   * [更新 TVM 官网](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E6%9B%B4%E6%96%B0-tvm-%E5%AE%98%E7%BD%91)
   * [发布公告](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E5%8F%91%E5%B8%83%E5%85%AC%E5%91%8A)
   * [补丁发布](https://tvm.hyper.ai/docs/about/contribute/Release_Process#%E8%A1%A5%E4%B8%81%E5%8F%91%E5%B8%83)
* [错误处理指南](https://tvm.hyper.ai/docs/about/contribute/error_handling-guide)
   * [在 C++ 中抛出特定错误](https://tvm.hyper.ai/docs/about/contribute/error_handling-guide#%E5%9C%A8-c-%E4%B8%AD%E6%8A%9B%E5%87%BA%E7%89%B9%E5%AE%9A%E9%94%99%E8%AF%AF)
   * [如何选择错误类型](https://tvm.hyper.ai/docs/about/contribute/error_handling-guide#%E5%9C%A8-c-%E4%B8%AD%E6%8A%9B%E5%87%BA%E7%89%B9%E5%AE%9A%E9%94%99%E8%AF%AF)


