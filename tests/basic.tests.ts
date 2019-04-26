import { run } from "../src/index";

describe("correct python is imported", () => {
  test("version is 3", () => {
    const version = run(`import sys; sys.version_info[0]`);
    expect(version).toEqual(3);
  });
});

describe("can run python and return a value", () => {
  test("string", () => {
    const x = run(`"Hello world!"`);
    expect(x).toEqual("Hello world!");
  });

  test("number", () => {
    const x = run(`123`);
    expect(x).toEqual(123);
  });

  test("list", () => {
    const x = run(`[1, 2, 3]`);
    expect(x).toEqual([1, 2, 3]);
  });

  test("tuple", () => {
    const x = run(`(1, 2, 3)`);
    expect(x).toEqual([1, 2, 3]);
  });

  test.skip("numpy array", () => {
    const x = run(`
import numpy as np;
np.array([1,2,3])`);

    expect(x).toEqual([1, 2, 3]);
  });

  test("lambda", () => {
    const x = run(`lambda x: x`);
    expect(x).toBeDefined();
    expect(x(123)).toBe(123);
  });

  test("function", () => {
    run(`def a(x):
    return x
`);
    const a = run(`a`);
    // const x = run(`a`);
    expect(a).toBeDefined();
    console.log("a is ", a);
    expect(a(123)).toBe(123);
  });
});
