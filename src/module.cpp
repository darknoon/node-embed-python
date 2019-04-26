#include <Python.h>
#include <napi.h>

#include <iostream>

#include "py_object_wrapper.h"
#include "utils.h"

// from utils.cc in python.node
class PyThreadStateLock {
 public:
  PyThreadStateLock(void) { py_gil_state = PyGILState_Ensure(); }

  ~PyThreadStateLock(void) { PyGILState_Release(py_gil_state); }

 private:
  PyGILState_STATE py_gil_state;
};

// Run arbitrary python code.
// Argument 0: string to evaluate
Napi::Value run(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto arg_count = info.Length();
  auto str = std::string(info[0].As<Napi::String>());

  // std::cout << "Will run: `" << str << "`" << std::endl;

  str = str + "\n";

  Napi::EscapableHandleScope scope(env);

  auto flags = PyCompilerFlags{Py_InspectFlag | Py_InteractiveFlag |
                               Py_BytesWarningFlag | Py_DebugFlag};

  // from Python/pythonrun.c@PyRun_SimpleStringFlags(...)
  PyObject* m = PyImport_AddModule("__main__");
  PyObject* globals = PyModule_GetDict(m);

  // description of Py_eval_input brought to you by
  // http://boost.cppll.jp/HEAD/libs/python/doc/tutorial/doc/using_the_interpreter.html
  // When using Py_eval_input, the input string must contain a single expression
  // and its result is returned. When using Py_file_input, the string can
  // contain an abitrary number of statements and None is returned.
  // Py_single_input works in the same way as Py_file_input but only accepts a
  // single statement.
  PyObject* result =
      PyRun_StringFlags(str.c_str(), Py_eval_input, globals, globals, &flags);

  // Debug logging of returned result type & ptr
  // std::cout << "> " << (result ? result->ob_type->tp_name : "") << "(" <<
  // result << ")" << std::endl;

  if (result == nullptr) {
    std::cout << "Attempting to handle python error" << std::endl;
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
  }
  auto retval = ConvertToJS(env, result);
  Py_XDECREF(result);
  return scope.Escape(retval);
}

// Initialize this module, setting up python
Napi::Object Init(Napi::Env env, Napi::Object exports) {
  Py_Initialize();
// We crash without this becaus the dylib isn't open yet
std:
  // import_array() defines some things inline, which conficts with  our Object
  // definition
  do_numpy_import();

  // PyObject* np = PyImport_ImportModule("numpy");
  // std::cout << "numpy is " << np << std::endl;
  // Py_XINCREF(np);
  exports.Set("run", Napi::Function::New(env, run));
  auto noexport = Napi::Object();
  PyObjectProxyHandler::Init(env, noexport);
  return exports;
}

NODE_API_MODULE(Python, Init);
