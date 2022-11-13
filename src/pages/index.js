import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.scss';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.indexCtas}>
          <Link className="button button--lg button--primary" to="/docs">
            查看文档
          </Link>
        </div>
        <div className={styles.rightAlignDetails}>
          <p className="subHeadingTwo">TVM 的全称为 Tensor Virtual Machine ，意为“向量虚拟机”。利用 TVM，工程师可以在任意硬件后端高效地优化和运行计算。本站由 MLC.AI 志愿者组织进行更新，由超神经 Hyper.AI 进行维护及托管。</p>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.tagline}
      description={siteConfig.title}>
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
