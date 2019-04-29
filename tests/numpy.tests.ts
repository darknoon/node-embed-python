import { exec, evalExpr } from "../src/index";

describe("can handle nested numpy arrays", () => {
  test("numpy array", () => {
    const { a } = exec(`
import numpy as np
a = np.array([[10, 20, 40]])
`);
    const data = a.tolist();
    expect(data).toEqual([[10, 20, 40]]);
  });

  test("likes big arrays", () => {
    const { a } = exec(`
import numpy as np
a = np.zeros((32, 32, 3))
`);
    const data = a.tolist();
    expect(data).toEqual([[10, 20, 40]]);
  });

  test("doesn't crash on ArrayBuffer", () => {
    const { np } = exec(`import numpy as np`);

    expect(() => {
      np.frombuffer(new ArrayBuffer(0));
    }).not.toThrow();
  });
});
