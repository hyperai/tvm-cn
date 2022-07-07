const path = require('path');

module.exports = function () {
  return {
    name: 'typekit-docusaurus-plugin',
    getClientModules() {
      return [path.resolve(__dirname, './attach-typekit')];
    },
  };
};
