// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Apache TVM 中文站',
  tagline: 'Apache TVM 是一个端到端的深度学习编译框架，适用于 CPU、GPU 和各种机器学习加速芯片。',
  url: 'https://tvm.hyper.ai',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.png',
  organizationName: 'hyperai', // Usually your GitHub org/user name.
  projectName: 'tvm-cn', // Usually your repo name.

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
  },

  // https://docusaurus.io/docs/markdown-features/code-blocks#interactive-code-editor
  themes: ['@docusaurus/theme-live-codeblock'],

  presets: [
    [
      '@docusaurus/preset-classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: 'https://github.com/hyperai/tvm-cn/edit/master/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          lastVersion: 'current',
          versions: {
            current: {
              label: '0.12.0',
            },
          }
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/edit/main/website/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/app.scss'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'TVM 中文站',
        hideOnScroll: false,
        logo: {
          alt: 'TVM Logo',
          src: 'img/favicon-dark.svg',
          srcDark: 'img/favicon.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'index',
            position: 'left',
            label: '查看文档',
          },
          // {to: '/docs', label: '查看文档', position: 'left'},
          {to: '/about', label: '关于', position: 'left'},
          {href: 'https://github.com/hyperai/tvm-cn', label: 'GitHub 项目', position: 'left'},
          {href: 'https://hyper.ai', label: '返回超神经', position: 'left'},

          // https://github.com/facebook/docusaurus/blob/main/website/docusaurus.config.js#L535C1-L535C1
          {
            type: 'docsVersionDropdown',
            position: 'right',
            dropdownActiveClassDisabled: true,
          },
        ],
      },
      footer: {
        style: 'light',
        links: [
        ],
        copyright: `© ${new Date().getFullYear()} Apache Software Foundation and Hyper.AI for Chinese Simplified mirror`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
      algolia: {
        apiKey: 'f36b719e2245a23ecd89c7e9a41937f2',
        indexName: 'docs',

        // Optional: see doc section below
        contextualSearch: true,

        // Optional: see doc section below
        appId: 'KU6TD2KAGA',

        // Optional: Algolia search parameters
        searchParameters: {
          attributesToSnippet: [
            'content:50'
          ],
        },

        //... other Algolia params
      },
    }),

  plugins: [
    'docusaurus-plugin-sass',
    require.resolve('./src/plugins/typekit/'),
  ],
  scripts: [
    {
      src: 'https://get.sparanoid.net/app.js',
      async: true,
      defer: true,
      'data-website-id': 'ba968118-b7e7-42b6-a445-977dad89216c'
    }
  ],

};

module.exports = config;
