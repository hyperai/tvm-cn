import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.scss';

import AboutImage from '@site/static/img/features/about-image.svg';
import AboutImageSmall from '@site/static/img/features/about-responsive-image.svg';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: '最佳性能',
    Svg: require('@site/static/img/features/speed.svg').default,
    description: (
      <>
        <p>通过编译和最小化运行时（Runtime），在现有硬件上优化机器学习工作负载，进而发挥最佳性能。</p>
      </>
    ),
  },
  {
    title: '支持任意硬件',
    Svg: require('@site/static/img/features/run.svg').default,
    description: (
      <>
        <p>可在 CPU、GPU、浏览器、微控制器、FPGA 等硬件运行</p>
        <p>同时可自动在多种后端生成和优化张量算子</p>
      </>
    ),
  },
  {
    title: '灵活可用',
    Svg: require('@site/static/img/features/Flexibility.svg').default,
    description: (
      <>
        <p>TVM 的灵活设计支持对区块稀疏度、量化（1、2、4、8 位整数，posit）、随机森林/经典 ML、内存规划、MISRA-C 兼容性、Python 原型设计。</p>
      </>
    ),
  },
  {
    title: '简单易用',
    Svg: require('@site/static/img/features/use.svg').default,
    description: (
      <>
        <p>在 Keras、MXNet、PyTorch、Tensorflow、CoreML、DarkNet 等的深度学习模型上编译。用 Python 借助 TVM 进行编译，次日即可用 C++、Rust 或 Java 建立生产堆栈。</p>
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={`${styles.blockWrap} ${clsx('col col--6')}`}>
      <div className={styles.block}>
        <div>
          <Svg className={styles.featureSvg} role="img" />
        </div>
        <div>
          <h3>{title}</h3>
          {description}
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <>
      <div className={styles.aboutSec}>
        <section className={`container`}>
          <ul className={styles.aboutInner}>
            <li className={styles.aboutImgCol}>
              <AboutImage className={styles.aboutImage} />
              <AboutImageSmall className={styles.aboutImageSmall} />
            </li>
            <li className={styles.aboutDetailsCol}>
              <h3 id="about-apache-tvm">About Apache TVM</h3>
              <p>Apache TVM 是一个正在 Apache 软件基金会（ASF）孵化的开源项目。我们致力于维护一个由机器学习、编译器和系统架构方面的专家及从业者组成的多样化社区，建立一个可访问、可扩展及自动化的开源框架，为任意硬件平台优化当前和新兴的机器学习模型。TVM 提供以下主要功能：</p>

              <ul>
                <li>将深度学习模型编译成最小可部署的模块。</li>
                <li>在更多的后端自动生成和优化模型的基础设施，进一步提高性能。</li>
              </ul>
            </li>
          </ul>
        </section>
      </div>

      <section className={styles.container}>
        <h2 className={styles.title}>主要功能及特点</h2>
      </section>
      <section className={styles.features}>
        <div className={styles.container}>
          <div className="row">
            {FeatureList.map((props, idx) => (
              <Feature key={idx} {...props} />
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
