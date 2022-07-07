import React from "react";
import CodeBlock from '@theme/CodeBlock';
import Tooltip from '@site/src/components/Tooltip';;

import css from "./styles.module.scss";
import Arrow from "./arrow-alt-circle-down-solid.svg";

const ServiceDemo = (props) => {
  const {
    original,
    replacement,
    meta,
    config,
    headingLevel,
    children
  } = props;
  const HtmlTag = headingLevel || 'h2';

  return (
    <>
      {/* https://github.com/facebook/docusaurus/issues/4029 */}
      {meta?.tier && (
        <>
          <HtmlTag>Service Tier</HtmlTag>
          <p className={css.tierWrap}>
            {meta?.tier.includes('beta') && (
              <Tooltip content="This service is in beta. It may be changed or termintated in the furtue">
                <span className="badge">Beta</span>
              </Tooltip>
            )}

            {meta?.tier.includes('paid') && (
              <Tooltip content="This is a paid service">
                <span className="badge badge--primary">Paid Service</span>
              </Tooltip>
            )}

            {meta?.tier.includes('cache') && (
              <Tooltip content="This service can help you cache the result from the response to reduce the cost of the original service (Up to 95%).">
                <span className="badge badge--info">Response Cache</span>
              </Tooltip>
            )}

            {meta?.tier.includes('stale') && (
              <Tooltip content="This service is in stale mode because no clients are currently using it">
                <span className="badge badge--warning">Stale</span>
              </Tooltip>
            )}

            {meta?.tier.includes('unstable') && (
              <Tooltip content="This service is unstable due to the way it implements or the SLA is low from the upstreams. So it may be changed or termintated in the furtue">
                <span className="badge badge--danger">Unstable</span>
              </Tooltip>
            )}
          </p>
        </>
      )}

      {/* This type of heading cannot be exported as ToC at the moment
        Ref: https://github.com/facebook/docusaurus/issues/3915
      */}

      {original && (
        <>
          <HtmlTag>Instructions</HtmlTag>
          <div className={css.intro}>
            <div className={`${css.introOriginal} ${replacement && css.hasAttachment}`}>
              <CodeBlock className={`language-yaml`}>
                {original}
              </CodeBlock>
            </div>

            {replacement && (
              <div className={css.decoratorAttachment}>
                <Arrow className={css.decorator} />
                <div className={`${css.introNew}`}>
                  <CodeBlock className={`language-yaml`}>
                    {replacement}
                  </CodeBlock>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {config && (
        <>
          <HtmlTag>Server Configurations</HtmlTag>
          <CodeBlock className="language-yaml">{config}</CodeBlock>
        </>
      )}

      {children && (
        <>
          <HtmlTag>Usage</HtmlTag>
          {children}
        </>
      )}
    </>
  )
};

export default ServiceDemo;
