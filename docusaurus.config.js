// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Apache TVM 中文文档',
  tagline: '一个用于 CPU、GPU 和加速器的端到端机器学习编译器框架',
  url: 'https://tvm.hyper.ai',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.png',
  organizationName: 'hyperai', // Usually your GitHub org/user name.
  projectName: 'tvm-cn', // Usually your repo name.

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
          // editUrl: 'https://github.com/facebook/docusaurus/edit/main/website/',
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
        title: 'TVM 中文文档',
        hideOnScroll: false,
        logo: {
          alt: 'TVM Logo',
          src: 'img/favicon-dark.svg',
          srcDark: 'img/favicon.svg',
        },
        items: [
          // {
          //   type: 'doc',
          //   docId: 'intro',
          //   position: 'left',
          //   label: 'Services',
          // },
          {to: '/docs', label: '查看文档', position: 'left'},
        ],
      },
      footer: {
        style: 'light',
        links: [
        ],
        copyright: `© ${new Date().getFullYear()} Apache Software Foundation and Hyper.AI for Chinese Simplified translation`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
      algolia: {
        apiKey: '4c79bcac69eb355e75986be320a047f2',
        indexName: 'docs',

        // Optional: see doc section below
        contextualSearch: true,

        // Optional: see doc section below
        appId: 'P21ECHP46V',

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
