/*
 * We support this case (strip 2 spaces)
 * const myFn = () => {
 *   const {a} = py`
 *   def a(x):
 *     return x + 1`
 * }
 *
 * Don't care about this case (awkward)
 * const myFn = () => {
 *   const {a} = py`def a(x):
 *     return x + 1`
 * }
 *
 * const myFn = () => {
 *   const {a} = py`def a(x):
 *     return x + 1`
 * }
 */

// Do some simple preprocessing on the input string to avoid having to do it in c++
const dedent = (strs, interps) => {
  if (typeof strs != "string" && strs.length > 1) {
    throw Error(
      `Dedent: string interpolation not allowed.
      ${strs.length} strings received.`
    );
  }
  let str = typeof strs == "string" ? strs : strs[0];
  // Split string into lines
  const lines = str.split("\n");
  if (lines.length == 1) {
    return str;
  } else {
    // Search for a non-empty line to base our indent off of
    let stripStr = "";
    let stripIndex = 0;
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i];
      const start = line.search(/\w/);
      if (start > 0) {
        stripIndex = start;
        stripStr = line.substring(0, stripIndex);
        break;
      }
    }
    return lines
      .map(l => {
        const lineStart = l.substring(0, stripIndex);
        // Make sure we are actually stripping expected whitespace and not content
        return lineStart === stripStr ? l.substring(stripIndex) : l;
      })
      .join("\n");
  }
};

module.exports.default = dedent;
