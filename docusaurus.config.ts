import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
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

  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          editUrl: 'https://github.com/hyperai/tvm-cn/edit/master/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          lastVersion: 'current',
          versions: {
            current: {
              label: '0.13.0',
            },
          },
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/edit/main/website/blog/',
        },
        theme: {
          customCss: './src/css/app.scss',
        },
        gtag: {
          trackingID: 'G-YY2E0ZQRP8',
          anonymizeIP: false,
        }
      } satisfies Preset.Options,
    ],
  ],

  // https://docusaurus.io/docs/markdown-features/math-equations
  stylesheets: [
    {
      // href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      // href: 'https://experiments-hk.sparanoid.net/jsd/npm/katex@0.13.24/dist/katex.min.css',
      href: 'https://workers.vrp.moe/api/jsd/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig: {
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
        {href: 'https://github.com/hyperai/tvm-cn', label: 'GitHub', position: 'left'},
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
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      // https://docusaurus.io/docs/migration/v3#prism-react-renderer-v20
      additionalLanguages: ['bash', 'diff', 'json', 'python'],
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
  } satisfies Preset.ThemeConfig,

  plugins: [
    'docusaurus-plugin-sass',
    './src/plugins/typekit/',
  ],
  scripts: [
    {
      src: 'https://get.openbayes.net/js/script.js',
      defer: true,
      'data-domain': 'tvm.hyper.ai'
    }
  ],

};

export default config
