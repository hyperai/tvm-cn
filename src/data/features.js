/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';

const FEATURES = [
  {
    title: {
      message: 'Powered by MDX',
      id: 'homepage.features.powered-by-mdx.title',
    },
    image: {
      src: '/img/undraw_typewriter.svg',
      width: 1009.54,
      height: 717.96,
    },
    text: (
      <>
        Save time and focus on text documents. Simply write docs and blog posts
        with MDX, and Docusaurus builds them into static HTML files ready to be
        served. You can even embed React components in your Markdown thanks to
        MDX.
      </>
    ),
  },
  {
    title: {
      message: 'Built Using React',
      id: 'homepage.features.built-using-react.title',
    },
    image: {
      src: '/img/undraw_react.svg',
      width: 1108,
      height: 731.18,
    },
    text: (
      <>
        Extend and customize your project&apos;s layout by writing React
        components. Leverage the pluggable architecture, and design your own
        site while reusing the same data created by Docusaurus plugins.
      </>
    ),
  },
  {
    title: {
      message: 'Ready for Translations',
      id: 'homepage.features.ready-for-translations.title',
    },
    image: {
      src: '/img/undraw_around_the_world.svg',
      width: 1137,
      height: 776.59,
    },
    text: (
      <>
        Localization comes out-of-the-box. Use git, Crowdin, or any other
        translation manager to translate your docs and deploy them individually.
      </>
    ),
  },
  {
    title: {
      message: 'Document Versioning',
      id: 'homepage.features.document-versioning.title',
    },
    image: {
      src: '/img/undraw_version_control.svg',
      width: 1038.23,
      height: 693.31,
    },
    text: (
      <>
        Support users on all versions of your project. Document versioning helps
        you keep documentation in sync with project releases.
      </>
    ),
  },
  {
    title: {
      message: 'Content Search',
      id: 'homepage.features.content-search.title',
    },
    image: {
      src: '/img/undraw_algolia.svg',
      width: 1137.97,
      height: 736.21,
    },
    text: (
      <>
        Make it easy for your community to find what they need in your
        documentation. We proudly support Algolia documentation search.
      </>
    ),
  },
];

export default FEATURES;
