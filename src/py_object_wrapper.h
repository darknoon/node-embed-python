#pragma once
#include <Python.h>
#include <napi.h>

/*
 * This is to enable passing wrapped objects back to JS from Python.
 *
 * For simple objects, a direct translation to JS types is applied, ie
 * String -> string
 * Dictionary -> Object
 *
 * For more complex types like numpy arrays, we must hold onto a handle to the
 * python object and make a js Proxy with this as the handler (see .cpp file for
 * details)
 */

class PyObjectProxyHandler : public Napi::ObjectWrap<PyObjectProxyHandler> {
 public:
  // Initialize
  static Napi::Object Initialize(Napi::Env env, Napi::Object exports);

  // Wrap an object
  static Napi::Value WrapObject(Napi::Env env, PyObject* object);

  // No need for constructor externally
  PyObjectProxyHandler(const Napi::CallbackInfo& info,
                       const PyObject* object = nullptr);

  // Check if provided value is a proxy to a PyObject external value
  static bool IsProxyValue(Napi::Value o);
  static Napi::External<PyObject> GetProxyValue(Napi::Value o);

 private:
  static Napi::FunctionReference proxyHandler;

  static Napi::Symbol secretHandshake;

  // Implementation of Proxy handlers interface
  static Napi::Value get(const Napi::CallbackInfo& info);
  static Napi::Value toString(const Napi::CallbackInfo& info);
  static void set(const Napi::CallbackInfo& info);
};

// Wrap a python function
Napi::Function WrapCallable(const Napi::Env& env, PyObject* fn);

// Callback for when you actually call a python function
Napi::Value CallPythonFunction(const Napi::CallbackInfo& info);
