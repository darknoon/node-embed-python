#pragma once
#include <Python.h>
#include <napi.h>

/*
 * This is to enable passing objects back to JS from Python.
 *
 * For simple objects, a direct translation to JS types is applied, ie
 * String -> string
 * Dictionary -> Object
 *
 * For more complex types like numpy arrays, we must hold onto a handle to the
 * python object and make a js Proxy with this as the handler (see .cpp file for
 * details):
 */

class PyObjectProxyHandler : public Napi::ObjectWrap<PyObjectProxyHandler> {
 public:
  // Init once
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  // Constructor
  PyObjectProxyHandler(const Napi::CallbackInfo& info,
                       const PyObject* object = nullptr);

  static Napi::Function WrapCallable(const Napi::Env& env, PyObject* fn);

 private:
  static Napi::FunctionReference proxyHandler;

  // Implementation of Proxy handlers interface
  static Napi::Value get(const Napi::CallbackInfo& info);
  static void set(const Napi::CallbackInfo& info);
};
