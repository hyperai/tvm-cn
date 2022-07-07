/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import RequestDemo from '@site/src/components/RequestDemo';

// https://docusaurus.io/docs/markdown-features/code-blocks#imports
// Add react-live imports you need here
const ReactLiveScope = {
  React,
  ...React,
  RequestDemo,
};

export default ReactLiveScope;
