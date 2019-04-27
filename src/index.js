"use strict";

// This will search for the native node-embed-python.node using the bindings library
const m = require("bindings")("node-embed-python");

// Export everything from the module
module.exports = m;
