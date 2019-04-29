"use strict";
// This will search for the native node-embed-python.node using the bindings library
const { exec, expr } = require("bindings")("node-embed-python");
const dedent = require("./dedent").default;

// Export dedent fn for testing, maybe has a use?
module.exports.dedent = dedent;
module.exports.exec = exec;
module.exports.expr = expr;

const py = (strs, ...rest) => {
  const str = dedent(strs, ...rest);
  return exec(str);
};
// Export everything from the module
module.exports.py = py;
