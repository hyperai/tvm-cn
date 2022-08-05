/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  /*
  tutorialSidebar: [
    // The original sidebar from docusaurus
    {type: 'autogenerated', dirName: '.'}
  ],
  */

  // But you can create a sidebar manually
  tutorialSidebar: [
    "index",
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        {type: 'doc', id: 'install/index'},
        {
          type: 'category',
          label: '贡献指南',
          link: {
              type:'doc', id: 'contribute/index'
          },
          items: [
            {type: 'autogenerated', dirName: 'contribute'}
          ]
        },
      ],
      collapsible: false
    },
    {
      type: 'category',
      label: 'User Guide',
      items: [
        {
          type: 'category',
          label: '用户教程',
          link: {
              type:'doc', id: 'user_guide/index'
          },
          items: [
            {type: 'autogenerated', dirName: 'user_guide/user_tutorial'}
          ]
        },
        {
          type: 'category',
          label: '用户指南',
          link: {
              type:'doc', id: 'user_guide/how_to'
          },
          items: [
            {type: 'autogenerated', dirName: 'user_guide/how_to_guide'}
          ]
        }
      ],
      collapsible: false
    },
    {
      type: 'category',
      label: 'Developer Guide',
      items: [
        {
          type: 'category',
          label: '开发者教程',
          link: {
              type:'doc', id: 'dev/index'
          },
          items: [
            {type: 'autogenerated', dirName: 'dev/tutorial'}
          ]
        },
        {
          type: 'category',
          label: '开发者指南',
          link: {
              type:'doc', id: 'dev/how_to'
          },
          items: [
            {type: 'autogenerated', dirName: 'dev/how_to'}
          ]
        }
      ],
      collapsible: false
    },
    {
      type: 'category',
      label: 'Architecture Guide',
      items: [
        {type: 'doc', id: 'arch/index'},
      ],
      collapsible: false
    },
    {
      type: 'category',
      label: 'Topic Guides',
      items: [
        {type: 'doc', id: 'topic/microtvm/index'},
        {type: 'doc', id: 'topic/vta/index'},
      ],
      collapsible: false
    }
  ]
};

module.exports = sidebars;
