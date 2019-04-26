module.exports = {
  roots: ["<rootDir>/src", "<rootDir>/tests"],
  transform: {
    "^.+\\.tsx?$": "ts-jest"
  },
  testRegex: "(/tests/.*|(\\.|/)(test|spec))\\.(js|ts|tsx|mjs)$",
  moduleDirectories: ["node_modules", "build/Debug"],
  moduleFileExtensions: ["ts", "tsx", "js", "jsx", "json", "node"]
};
