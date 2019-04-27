#pragma once

#include <Python.h>
#include <napi.h>

int do_numpy_import(void);

Napi::Value ConvertToJSException(const Napi::Env& env, PyObject* py_exception);

Napi::String StringToJS(const Napi::Env& env, PyObject* py_object);

Napi::Value ConvertToJS(const Napi::Env& env, PyObject* py_object);

PyObject* ConvertToPy(Napi::Value value);

class PyThreadStateLock {
 public:
  PyThreadStateLock(void) { py_gil_state = PyGILState_Ensure(); }

  ~PyThreadStateLock(void) { PyGILState_Release(py_gil_state); }

 private:
  PyGILState_STATE py_gil_state;
};
