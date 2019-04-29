/**
 * exec runs a sequence of python statements, returning a mapping of
 * the variables defined within them, eg:
 *
 * exec(`a = 123`) => {a: 123}
 */
export function exec(str: string): { [variableName: string]: any };

/**
 * py template literal
 * @param str Python
 * @description
 * python-node-embed also supports the py`...` template literal for multi-line statements
 * ```
 * const {code} = py`code = "here"`
 * ```
 */
export function py(
  str: TemplateStringsArray,
  ...any: any[]
): { [variableName: string]: any };

/**
 * expr runs a single Python expression, returning the result
 * expr("1 + 2") => 3
 * expr("[1,2] + [3]") => [1,2,3]
 */
export function expr(str: string): any;
