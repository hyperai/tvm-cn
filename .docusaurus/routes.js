import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', 'ab7'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', 'bf2'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'f86'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', '1cc'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '120'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '42d'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', 'ca1'),
    exact: true
  },
  {
    path: '/enterprise',
    component: ComponentCreator('/enterprise', '8d4'),
    exact: true
  },
  {
    path: '/markdown-page',
    component: ComponentCreator('/markdown-page', '2b0'),
    exact: true
  },
  {
    path: '/search',
    component: ComponentCreator('/search', 'a5b'),
    exact: true
  },
  {
    path: '/updates',
    component: ComponentCreator('/updates', 'e14'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '26c'),
    routes: [
      {
        path: '/docs/',
        component: ComponentCreator('/docs/', 'a8c'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/',
        component: ComponentCreator('/docs/arch/', 'c08'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/benchmark',
        component: ComponentCreator('/docs/arch/benchmark', 'eb5'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/convert_layout',
        component: ComponentCreator('/docs/arch/convert_layout', '4a8'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/debugger',
        component: ComponentCreator('/docs/arch/debugger', '1f0'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/device_target_interactions',
        component: ComponentCreator('/docs/arch/device_target_interactions', '0db'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/frontend/tensorflow',
        component: ComponentCreator('/docs/arch/frontend/tensorflow', 'c79'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/hybrid_script',
        component: ComponentCreator('/docs/arch/hybrid_script', '938'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/inferbound',
        component: ComponentCreator('/docs/arch/inferbound', '489'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/introduction_to_module_serialization',
        component: ComponentCreator('/docs/arch/introduction_to_module_serialization', '17a'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/microtvm_design',
        component: ComponentCreator('/docs/arch/microtvm_design', '9e8'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/microtvm_project_api',
        component: ComponentCreator('/docs/arch/microtvm_project_api', '418'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/model_library_format',
        component: ComponentCreator('/docs/arch/model_library_format', 'add'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/pass_infra',
        component: ComponentCreator('/docs/arch/pass_infra', '725'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/relay_intro',
        component: ComponentCreator('/docs/arch/relay_intro', 'c18'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/relay_op_strategy',
        component: ComponentCreator('/docs/arch/relay_op_strategy', '750'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/runtime',
        component: ComponentCreator('/docs/arch/runtime', '1bf'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/runtimes/vulkan',
        component: ComponentCreator('/docs/arch/runtimes/vulkan', '0de'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/security',
        component: ComponentCreator('/docs/arch/security', '4bb'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/arch/virtual_machine',
        component: ComponentCreator('/docs/arch/virtual_machine', 'b1e'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/contribute/',
        component: ComponentCreator('/docs/contribute/', '81a'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/contribute/ci',
        component: ComponentCreator('/docs/contribute/ci', 'cdc'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/contribute/code_guide',
        component: ComponentCreator('/docs/contribute/code_guide', '1c6'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/contribute/code_review',
        component: ComponentCreator('/docs/contribute/code_review', '0c0'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/contribute/committer_guide',
        component: ComponentCreator('/docs/contribute/committer_guide', 'd33'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/contribute/community',
        component: ComponentCreator('/docs/contribute/community', '2de'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/contribute/document',
        component: ComponentCreator('/docs/contribute/document', 'd2b'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/contribute/error_handling',
        component: ComponentCreator('/docs/contribute/error_handling', '31e'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/contribute/git_howto',
        component: ComponentCreator('/docs/contribute/git_howto', 'e57'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/contribute/pull_request',
        component: ComponentCreator('/docs/contribute/pull_request', 'd31'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/contribute/release_process',
        component: ComponentCreator('/docs/contribute/release_process', '860'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/dev/how_to/',
        component: ComponentCreator('/docs/dev/how_to/', 'e45'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/dev/how_to/debugging_tvm',
        component: ComponentCreator('/docs/dev/how_to/debugging_tvm', 'beb'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/dev/how_to/pytest_target_parametrization',
        component: ComponentCreator('/docs/dev/how_to/pytest_target_parametrization', '341'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/dev/how_to/relay_add_op',
        component: ComponentCreator('/docs/dev/how_to/relay_add_op', '0d3'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/dev/how_to/relay_add_pass',
        component: ComponentCreator('/docs/dev/how_to/relay_add_pass', 'b53'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/dev/how_to/relay_bring_your_own_codegen',
        component: ComponentCreator('/docs/dev/how_to/relay_bring_your_own_codegen', 'b14'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/dev/tutorial/',
        component: ComponentCreator('/docs/dev/tutorial/', '2fc'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/dev/tutorial/codebase_walkthrough',
        component: ComponentCreator('/docs/dev/tutorial/codebase_walkthrough', '88c'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/',
        component: ComponentCreator('/docs/how_to/', '980'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/deploy/',
        component: ComponentCreator('/docs/how_to/deploy/', '1cb'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/deploy/android',
        component: ComponentCreator('/docs/how_to/deploy/android', '1d0'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/deploy/arm_compute_lib',
        component: ComponentCreator('/docs/how_to/deploy/arm_compute_lib', '475'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/deploy/bnns',
        component: ComponentCreator('/docs/how_to/deploy/bnns', '471'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/deploy/cpp_deploy',
        component: ComponentCreator('/docs/how_to/deploy/cpp_deploy', '6ff'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/deploy/hls',
        component: ComponentCreator('/docs/how_to/deploy/hls', '67c'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/deploy/integrate',
        component: ComponentCreator('/docs/how_to/deploy/integrate', '061'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/deploy/tensorrt',
        component: ComponentCreator('/docs/how_to/deploy/tensorrt', 'b38'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/deploy/vitis_ai',
        component: ComponentCreator('/docs/how_to/deploy/vitis_ai', '25b'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/profile/',
        component: ComponentCreator('/docs/how_to/profile/', 'ae4'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/how_to/profile/papi',
        component: ComponentCreator('/docs/how_to/profile/papi', '3c3'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/install/',
        component: ComponentCreator('/docs/install/', '26c'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/install/docker',
        component: ComponentCreator('/docs/install/docker', '73b'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/install/from_source',
        component: ComponentCreator('/docs/install/from_source', 'ba5'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/install/nnpack',
        component: ComponentCreator('/docs/install/nnpack', '5f4'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/install/tlcpack',
        component: ComponentCreator('/docs/install/tlcpack', '44d'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/links',
        component: ComponentCreator('/docs/reference/api/links', 'b3b'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/',
        component: ComponentCreator('/docs/reference/api/python/', 'c9d'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/auto_scheduler',
        component: ComponentCreator('/docs/reference/api/python/auto_scheduler', '3ec'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/autotvm',
        component: ComponentCreator('/docs/reference/api/python/autotvm', '6e3'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/contrib',
        component: ComponentCreator('/docs/reference/api/python/contrib', 'acf'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/driver',
        component: ComponentCreator('/docs/reference/api/python/driver', 'c2e'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/error',
        component: ComponentCreator('/docs/reference/api/python/error', 'eae'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/graph_executor',
        component: ComponentCreator('/docs/reference/api/python/graph_executor', '4b7'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/ir',
        component: ComponentCreator('/docs/reference/api/python/ir', 'd71'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/micro',
        component: ComponentCreator('/docs/reference/api/python/micro', '560'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/ndarray',
        component: ComponentCreator('/docs/reference/api/python/ndarray', '376'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/relay/',
        component: ComponentCreator('/docs/reference/api/python/relay/', 'd24'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/relay/analysis',
        component: ComponentCreator('/docs/reference/api/python/relay/analysis', '7f4'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/relay/backend',
        component: ComponentCreator('/docs/reference/api/python/relay/backend', 'b5d'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/relay/dataflow_pattern',
        component: ComponentCreator('/docs/reference/api/python/relay/dataflow_pattern', '4bd'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/relay/frontend',
        component: ComponentCreator('/docs/reference/api/python/relay/frontend', '562'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/relay/image',
        component: ComponentCreator('/docs/reference/api/python/relay/image', '29a'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/relay/nn',
        component: ComponentCreator('/docs/reference/api/python/relay/nn', '4a3'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/relay/testing',
        component: ComponentCreator('/docs/reference/api/python/relay/testing', 'f83'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/relay/transform',
        component: ComponentCreator('/docs/reference/api/python/relay/transform', '1e6'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/relay/vision',
        component: ComponentCreator('/docs/reference/api/python/relay/vision', '9fb'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/rpc',
        component: ComponentCreator('/docs/reference/api/python/rpc', '2ab'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/runtime',
        component: ComponentCreator('/docs/reference/api/python/runtime', 'd5a'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/target',
        component: ComponentCreator('/docs/reference/api/python/target', '154'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/te',
        component: ComponentCreator('/docs/reference/api/python/te', '667'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/tir',
        component: ComponentCreator('/docs/reference/api/python/tir', '5a7'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/topi',
        component: ComponentCreator('/docs/reference/api/python/topi', '9f3'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/api/python/vta/',
        component: ComponentCreator('/docs/reference/api/python/vta/', 'a67'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/langref/',
        component: ComponentCreator('/docs/reference/langref/', '5c3'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/langref/hybrid_script',
        component: ComponentCreator('/docs/reference/langref/hybrid_script', 'f54'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/langref/relay_adt',
        component: ComponentCreator('/docs/reference/langref/relay_adt', 'b68'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/langref/relay_expr',
        component: ComponentCreator('/docs/reference/langref/relay_expr', '2f6'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/langref/relay_op',
        component: ComponentCreator('/docs/reference/langref/relay_op', '440'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/langref/relay_pattern',
        component: ComponentCreator('/docs/reference/langref/relay_pattern', 'cc5'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/langref/relay_type',
        component: ComponentCreator('/docs/reference/langref/relay_type', '167'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/reference/publications',
        component: ComponentCreator('/docs/reference/publications', '1ee'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/topic/microtvm/',
        component: ComponentCreator('/docs/topic/microtvm/', '8a5'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/topic/vta/',
        component: ComponentCreator('/docs/topic/vta/', '4ef'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/topic/vta/dev/',
        component: ComponentCreator('/docs/topic/vta/dev/', '42d'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/topic/vta/dev/config',
        component: ComponentCreator('/docs/topic/vta/dev/config', '0ea'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/topic/vta/dev/hardware',
        component: ComponentCreator('/docs/topic/vta/dev/hardware', '75c'),
        exact: true,
        sidebar: "tutorialSidebar"
      },
      {
        path: '/docs/topic/vta/install',
        component: ComponentCreator('/docs/topic/vta/install', 'de8'),
        exact: true,
        sidebar: "tutorialSidebar"
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', 'c65'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
