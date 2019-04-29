import { expr, exec, py } from "../src/index";

describe("correct python is imported", () => {
  test("version is 3", () => {
    const { sys } = exec(`import sys`);
    const version = sys.version_info[0];
    expect(version).toEqual(3);
  });
});

describe("can eval python and return a value", () => {
  test("string", () => {
    const x = expr(`"Hello world!"`);
    expect(x).toEqual("Hello world!");
  });

  test("number", () => {
    const x = expr(`123`);
    expect(x).toEqual(123);
  });

  test("list", () => {
    const x = expr(`[1, 2, 3]`);
    expect(x).toEqual([1, 2, 3]);
  });

  test("tuple", () => {
    const x = expr(`(1, 2, 3)`);
    expect(x).toEqual([1, 2, 3]);
  });

  test("None", () => {
    const x = expr(`None`);
    expect(x).toEqual(null);
  });

  test("numpy array", () => {
    const { np } = exec("import numpy as np");
    const { list } = exec("list = list");

    const x = np.array([1, 2, 3]);
    console.log("type of x is ", typeof x);
    const xl = list(x);
    expect(xl).toEqual([1, 2, 3]);
  });

  test("lambda", () => {
    const x = expr(`lambda x: x`);
    expect(x).toBeDefined();
    expect(x(123)).toBe(123);
  });

  test("function", () => {
    const { a } = py`
    def a(x):
        return x
    `;
    // const x = expr(`a`);
    expect(a).toBeDefined();
    expect(a(123)).toBe(123);
  });
});

describe("Handles Python exceptions", () => {
  test("simple raise", () => {
    expect(() => {
      exec(`raise Exception("Test Exception")`);
    }).toThrowError("Test Exception");
  });

  test("syntax error", () => {
    expect(() => {
      exec(`def myfun`);
    }).toThrowError(/SyntaxError: .*/);
  });
});
