import dedent from "../src/dedent";

test("handles single-line string", () => {
  expect(dedent("1 + 2")).toBe("1 + 2");
});

test("handles multi-line python", () => {
  const decl = dedent`
  def a(x):
    return x
  `;

  expect(decl).toBe("\ndef a(x):\n  return x\n");
});
