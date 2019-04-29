#pragma once

#include <Python.h>
#include <napi.h>

// TODO: move into some kind of extension system for interop with Python native
// modules
int do_numpy_import(void);

Napi::Error ConvertToJSException(Napi::Env env,
                                 PyObject* py_exception_type,
                                 PyObject* py_exception_value,
                                 PyObject* py_exception_traceback);

Napi::String StringToJS(const Napi::Env& env, PyObject* py_object);

Napi::Value ConvertToJS(const Napi::Env& env, PyObject* py_object);

std::string CoerceToString(Napi::Value v);

PyObject* ConvertToPy(Napi::Value value);

// Get the current Python exception, and throw it as a Javascript exception
void ThrowPythonException(Napi::Env env);

class PyThreadStateLock {
 public:
  PyThreadStateLock(void) { py_gil_state = PyGILState_Ensure(); }

  ~PyThreadStateLock(void) { PyGILState_Release(py_gil_state); }

 private:
  PyGILState_STATE py_gil_state;
};

std::string describe_symbol(Napi::Symbol symbol);

std::string describe_napi_value_type(Napi::Value value);