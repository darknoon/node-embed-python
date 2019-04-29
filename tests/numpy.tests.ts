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
});
