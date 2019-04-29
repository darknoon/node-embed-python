import { py } from "../src/index";

// Implement zeros / ones independently to verify correctness
function* repeat<V>(times: number, v: V) {
  for (let i = 0; i < times; i++) {
    yield v;
  }
}
const fill = (shape: Array<number>, value: number = 0) => {
  const shapeReversed = [...shape];
  shapeReversed.reverse();
  let cur: any = value;
  for (let dimSize of shapeReversed) {
    // Add a level of nesting
    cur = Array.from(repeat(dimSize, cur));
  }
  return cur;
};

describe("can handle nested numpy arrays", () => {
  test("numpy array", () => {
    const { a } = py`
    import numpy as np
    a = np.array([[10, 20, 40]])`;

    const data = a.tolist();
    expect(data).toEqual([[10, 20, 40]]);
  });

  test("likes big arrays", () => {
    const { a } = py`
    import numpy as np
    a = np.zeros((32, 8, 3))`;

    // Call numpy.ndarray's tolist method
    const data = a.tolist();
    // Construct our own zero-filled Array of Arrays of Arrays in JS
    const zeros = fill([32, 8, 3], 0);
    // Verify that they are deep-equal
    expect(data).toEqual(zeros);
  });

  test.skip("doesn't crash on ArrayBuffer", () => {
    const { np } = py`import numpy as np`;

    expect(() => {
      np.frombuffer(new ArrayBuffer(0));
    }).not.toThrow();
  });
});
