
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

// Make sure these static fields are real variables included in the binary
Napi::FunctionReference PyObjectProxyHandler::proxyHandler;
Napi::Symbol PyObjectProxyHandler::secretHandshake;

Napi::Object PyObjectProxyHandler::Initialize(Napi::Env env,
                                              Napi::Object exports) {
  Napi::Function func = DefineClass(env, "PyObjectHandler<?>",
                                    {
                                        StaticMethod("get", get),
                                        StaticMethod("set", set),
                                        StaticMethod("toString", toString),
                                    });

  proxyHandler = Napi::Persistent(func);
  proxyHandler.SuppressDestruct();

  secretHandshake = Napi::Symbol::New(env, "@@PyObject");

  return exports;
}

PyObjectProxyHandler::PyObjectProxyHandler(const Napi::CallbackInfo& info,
                                           const PyObject* object)
    : Napi::ObjectWrap<PyObjectProxyHandler>(info) {
  // TODO: do we need to do anything here?
}

bool PyObjectProxyHandler::IsProxyValue(Napi::Value value) {
  auto object = value.As<Napi::Object>();
  auto getHandshake = object.Get("@@secretHandshake@@");
  if (getHandshake.IsExternal()) {
    auto external =
        object.Get("@@secretHandshake@@").As<Napi::External<PyObject>>();
    return true;
  } else {
    return false;
  }
}

Napi::External<PyObject> PyObjectProxyHandler::GetProxyValue(
    Napi::Value value) {
  auto object = value.As<Napi::Object>();
  auto getHandshake = object.Get("@@secretHandshake@@");
  auto external =
      object.Get("@@secretHandshake@@").As<Napi::External<PyObject>>();
  return external;
}

Napi::Value PyObjectProxyHandler::WrapObject(Napi::Env env, PyObject* object) {
  assert(object);

  std::cout << "Wrapping object" << std::endl;
  // grab Proxy class
  auto Proxy = env.Global().Get("Proxy").As<Napi::Function>();
  // std::cout << "Proxy is: " << Proxy.ToString().Utf8Value() << std::endl;

  // Use an External as the target of the proxy (holds the PyObject for us to
  // use later)
  using Hint = int;
  // Deref Python object when this proxy's target gets cleaned up
  auto finalizer = [](Napi::Env _, PyObject* o, Hint* __) {
    printf("Finalized proxy for obj %p", o);
    Py_DECREF(o);
  };
  auto wrapped =
      Napi::External<PyObject>::New(env, object, finalizer, (Hint*)nullptr);
  Py_INCREF(object);
  // std::cout << "wrapped is: " << wrapped << std::endl;

  // Grab the global proxy handler
  auto handler = proxyHandler.Value();
  // std::cout << "handler is: " << handler.ToString().Utf8Value() << std::endl;

  // auto proxy =
  //     Proxy.New({napi_value(Napi::EscapableHandleScope(env).Escape(wrapped)),
  //                napi_value(Napi::EscapableHandleScope(env).Escape(handler))});

  auto proxy = Proxy.New({napi_value(wrapped), napi_value(handler)});

  // std::cout << "proxy is: " << proxy << std::endl;

  Napi::EscapableHandleScope scope(env);
  return scope.Escape(proxy);
}

Napi::Value PyObjectProxyHandler::get(const Napi::CallbackInfo& info) {
  auto env = info.Env();

  auto target = info[0];
  auto prop = info[1];
  auto receiver = info[2];

  if (prop.Type() == napi_string) {
    auto propname = prop.ToString().Utf8Value();
    PyObject* py_object = target.As<Napi::External<PyObject>>().Data();
    std::cout << "PyObject get " << py_object << "[" << propname << "]"
              << std::endl;

    PyObject* py_value = NULL;
    if (propname == "toString") {
      py_value = PyObject_Str(py_object);
    } else if (propname == "valueOf") {
      return env.Null();
    } else if (propname == "@@secretHandshake@@") {
      Napi::EscapableHandleScope scope(env);

      return scope.Escape(target);
    } else {
      PyObject* attr = ConvertToPy(prop);
      py_value = PyObject_GetAttr(py_object, attr);
      // std::cout << "py_object." << prop.ToString().Utf8Value() << " = " <<
      // obj
      //           << std::endl;
      Py_DECREF(attr);
    }

    Napi::EscapableHandleScope scope(env);
    auto js_obj = ConvertToJS(env, py_value);
    return scope.Escape(js_obj);
  } else if (prop.Type() == napi_number) {
    std::cout << "get[number] unimplemented" << std::endl;
    return env.Undefined();
  } else if (prop.Type() == napi_symbol) {
    // One of the special symbol gets in JS
    auto symbol = prop.As<Napi::Symbol>();

    std::cout << "get[" << describe_symbol(symbol) << "]" << std::endl;
    auto iteratorSymbol = Napi::Symbol::WellKnown(env, "iterator");
    auto toPrimitiveSymbol = Napi::Symbol::WellKnown(env, "toPrimitive");
    // return env.Null();
    if (symbol == secretHandshake) {
      Napi::EscapableHandleScope scope(env);
      return scope.Escape(target);
    } else {
      // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Proxy/handler/get
      // Reflect.get(target, prop, receiver)
      auto Reflect = env.Global().Get("Reflect").As<Napi::Object>();
      auto get = Reflect.Get("get").As<Napi::Function>();
      auto result = get.Call(Reflect, {target, prop, receiver});
      std::cout << "Reflect.get() -> " << CoerceToString(result) << " ("
                << describe_napi_value_type(result) << ")" << std::endl;

      // Escape it
      Napi::EscapableHandleScope scope(env);
      return scope.Escape(result);
    }
  } else {
    // TODO: throw here. Can't index by other things yet.
    Napi::EscapableHandleScope scope(env);
    return scope.Escape(Napi::Value());
  }
}
void PyObjectProxyHandler::set(const Napi::CallbackInfo& info) {
  std::cout << "set() handler called" << std::endl;
}

Napi::Value PyObjectProxyHandler::toString(const Napi::CallbackInfo& info) {
  auto env = info.Env();

  auto target = info[0];
  auto receiver = info[1];
  std::cout << "toString() on PyObject Proxy" << std::endl;

  return Napi::String::New(env, "<PyObject Proxy>");
}

// TODO: move this function wrapper
// For calling functions
Napi::Function WrapCallable(const Napi::Env& env, PyObject* fn) {
  Py_XINCREF(fn);
  // std::cout << "wrapping callable " << fn << std::endl;
  // TODO: set a better description for the python function
  return Napi::Function::New(env, &CallPythonFunction, "[[python call]]", fn);
}

Napi::Value CallPythonFunction(const Napi::CallbackInfo& info) {
  Napi::EscapableHandleScope scope(info.Env());
  auto d = (PyObject*)info.Data();
  auto length = info.Length();
  std::cout << "Trying to call callable " << d << " with " << length << " args."
            << std::endl;
  // Create args tuple for python
  auto args = PyTuple_New(length);
  for (int i = 0; i < length; i++) {
    auto arg = ConvertToPy(info[i]);
    if (!arg) {
      ThrowPythonException(info.Env());
    }
    // Steals reference to arg, so don't need to decref
    PyTuple_SetItem(args, i, arg);
  }

  auto kwargs = nullptr;
  auto result = PyObject_Call(d, args, kwargs);
  Py_XDECREF(args);

  std::cout << "result of call is " << result << " " << result->ob_type->tp_name
            << std::endl;

  auto js_value = ConvertToJS(info.Env(), result);
  // auto js_value_str = CoerceToString(js_value);
  // std::cout << "result of call in JS is " << js_value_str
  //           << " type: " << js_value.Type() << std::endl;

  Py_XDECREF(result);
  return scope.Escape(js_value);
}
