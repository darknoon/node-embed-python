
#include "py_object_wrapper.h"

#include <iostream>
#include "utils.h"

/*
NAPI decided not to expose the NamedGetter / IndexedGetter directly, so we will
need to create a JS Proxy that will lookup properties on the python object.

Pseudo-code / js interface

interface PyObjectProxyHandler {

  static ownKeys(target: PyObjectPtr) {
    <cpp: get python __getattr__ and then convert to js>
  }

  static get(target: PyObjectPtr, prop: string | Symbol) {
    <cpp: get python __getattr__ and then convert to js>
  }

  static set(target: PyObjectPtr, prop: string | Symbol, value) {
      <cpp: get python __setattr__ and then convert to js>
  }

}

const toJS = o => {
  if (o is PyObjectPtr) {
    return new Proxy(pyobj, Handler);
  } else if (o is list or o is tuple) {
    return new Array(o);
  } etc {
    ...
  }

}


 */

Napi::FunctionReference PyObjectProxyHandler::proxyHandler;

Napi::Object PyObjectProxyHandler::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "Wrapped<?>",
                                    {
                                        StaticMethod("get", get),
                                        StaticMethod("set", set),
                                    });

  proxyHandler = Napi::Persistent(func);
  proxyHandler.SuppressDestruct();
  return exports;
}

static Napi::Value CallPython(const Napi::CallbackInfo& info) {
  Napi::EscapableHandleScope scope(info.Env());
  auto d = (PyObject*)info.Data();
  auto length = info.Length();
  std::cout << "Trying to call callable " << d << " with " << length << " args."
            << std::endl;
  // Create args tuple for python
  auto args = PyTuple_New(length);
  for (int i = 0; i < length; i++) {
    auto arg = ConvertToPy(info[i]);
    // Steals reference to arg, so don't need to decref
    PyTuple_SetItem(args, i, arg);
  }

  auto kwargs = nullptr;
  auto result = PyObject_Call(d, args, kwargs);
  Py_XDECREF(args);

  std::cout << "result of call is " << result << " " << result->ob_type->tp_name
            << std::endl;

  auto js_value = ConvertToJS(info.Env(), result);
  Py_XDECREF(result);
  return scope.Escape(js_value);
}

Napi::Function PyObjectProxyHandler::WrapCallable(const Napi::Env& env,
                                                  PyObject* fn) {
  Py_XINCREF(fn);
  // std::cout << "wrapping callable " << fn << std::endl;
  return Napi::Function::New(env, &CallPython, "[[python call]]", fn);
}

PyObjectProxyHandler::PyObjectProxyHandler(const Napi::CallbackInfo& info,
                                           const PyObject* object)
    : Napi::ObjectWrap<PyObjectProxyHandler>(info) {
  auto env = info.Env();

  Napi::Function func = DefineClass(env, "Wrapped<?>",
                                    {
                                        StaticMethod("get", get),
                                        StaticMethod("set", set),
                                    });

  Napi::HandleScope scope(info.Env());
  Napi::String value = info[0].As<Napi::String>();
}

Napi::Value PyObjectProxyHandler::get(const Napi::CallbackInfo& info) {
  return Napi::Value();
}
void PyObjectProxyHandler::set(const Napi::CallbackInfo& info) {}
