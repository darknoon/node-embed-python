/**
 * exec runs a sequence of python statements, returning a mapping of
 * the variables defined within them, eg:
 *
 * exec(`a = 123`) => {a: 123}
 */
export function exec(str: string): { [variableName: string]: any };

/**
 * evalExpr runs a single Python expression, returning the result
 * evalExpr("1 + 2") => 3
 * evalExpr("[1,2] + [3]") => [1,2,3]
 */
export function evalExpr(str: string): any;
