import React, { useEffect, useState } from "react";
import { baseUrl } from '@site/src/components/utils.js';
import CodeBlock from '@theme/CodeBlock';

import css from "./styles.module.scss";

const RequestDemo = ({uri}) => {

  const [error, setError] = useState(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [result, setResult] = useState([]);

  useEffect(() => {
    fetch(`https://${baseUrl}${uri}`)
      .then(res => res.json())
      .then(
        (result) => {
          setIsLoaded(true);
          setResult(result);
        },
        (error) => {
          setIsLoaded(true);
          setError(error);
        }
      )
  }, [])

  if (error) {
    return <div>Error: {error.message}</div>;
  } else if (!isLoaded) {
    return <div>Loading...</div>;
  } else {
    return (
      <CodeBlock className={`${css.codeOutput} language-json`}>
        {JSON.stringify(result, null, 2)}
      </CodeBlock>
    );
  }

};

export default RequestDemo;
