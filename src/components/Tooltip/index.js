import React from "react";
import Tippy from '@tippyjs/react';
import { roundArrow } from 'tippy.js';
import 'tippy.js/dist/tippy.css';
import 'tippy.js/animations/shift-away.css';
import 'tippy.js/dist/svg-arrow.css';
import './index.scss';

const Tooltip = (props) => (
  <Tippy {...props} />
)

Tooltip.defaultProps = {
  arrow: roundArrow,
  animation: 'shift-away',
  delay: 150,
  trigger: 'mouseenter focus click',
  theme: 'light',
  // Avoid showing empty tippy instance
  // Ref https://github.com/atomiks/tippyjs/issues/526
  onShow: (options) => {
    return !!options.props.content.textContent;
  },
};

export default Tooltip;
