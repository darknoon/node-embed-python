#include <Python.h>
#include <napi.h>

#include <iostream>

#include "py_object_wrapper.h"
#include "utils.h"

/**
 * Run arbitrary python statements.
 * Returns any local variables defined in an object
 */
Napi::Value exec(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto str = std::string(info[0].As<Napi::String>());
  // Doesn't like parsing without newline, but we don't need to be so strict
  str = str + "\n";

  Napi::EscapableHandleScope scope(env);

  auto flags = PyCompilerFlags{Py_InspectFlag | Py_InteractiveFlag};

  PyObject* m = PyImport_AddModule("__main__");
  PyObject* globals = PyModule_GetDict(m);
  PyObject* locals = PyDict_New();

  // Py_file_input lets you run multiple statements
  // http://boost.cppll.jp/HEAD/libs/python/doc/tutorial/doc/using_the_interpreter.html
  // and https://stackoverflow.com/a/2220790
  // We want expressions to run but
  PyObject* result =
      PyRun_StringFlags(str.c_str(), Py_file_input, globals, locals, &flags);

  // Exception handling
  if (!result) {
    ThrowPythonException(env);
  }

  // Return a dictionary of local variables to the caller
  auto retval = ConvertToJS(env, locals);

  Py_DECREF(locals);
  Py_XDECREF(result);
  return scope.Escape(retval);
}

/**
 * Run a single python expression, eg "1+2", not "a=1"
 * Returns any local variables defined in an object
 */

Napi::Value evalExpr(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto str = std::string(info[0].As<Napi::String>());
  // Doesn't like parsing without newline, but we don't need to be so strict
  str = str + "\n";

  Napi::EscapableHandleScope scope(env);

  auto flags = PyCompilerFlags{Py_InspectFlag | Py_InteractiveFlag};

  PyObject* m = PyImport_AddModule("__main__");
  PyObject* globals = PyModule_GetDict(m);
  // Don't care about locals for eval
  PyObject* locals = PyDict_New();

  PyObject* result =
      PyRun_StringFlags(str.c_str(), Py_eval_input, globals, locals, &flags);
  // TODO: exception handling
  std::cout << "evalExpr() result value:" << result << std::endl;

  // Exception handling
  if (!result) {
    ThrowPythonException(env);
    return env.Undefined();
  }
  auto js_result = ConvertToJS(env, result);

  Py_DECREF(locals);
  Py_XDECREF(result);
  return scope.Escape(js_result);
}

// Initialize this module, setting up python
Napi::Object Init(Napi::Env env, Napi::Object exports) {
  std::cout << "Python initializing..." << std::endl;
  Py_Initialize();

  PyThreadStateLock py_thread_lock;
  std::cout << "Python initialized." << std::endl;
  // We crash without this becaus the dylib isn't open yet
  // import_array() defines some things inline, which conficts with  our Object
  // definition
  do_numpy_import();

  exports.Set("exec", Napi::Function::New(env, exec));
  exports.Set("evalExpr", Napi::Function::New(env, evalExpr));
  PyObjectProxyHandler::Initialize(env, exports);
  return exports;
}

NODE_API_MODULE(Python, Init);
