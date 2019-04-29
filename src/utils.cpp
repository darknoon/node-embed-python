#include "utils.h"

#include <iostream>
#include "py_object_wrapper.h"

// Silence some warnings. Don't need deprecated API.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

int do_numpy_import(void) {
  // This is a macro, so this function isn't useless
  // It actually returns an int from inside the macro
  import_array();
  // Silences warning
  return 0;
}

std::string CoerceToString(Napi::Value v) {
  auto env = v.Env();
  napi_value result;
  napi_status status =
      napi_get_named_property(env, napi_value(v), "toString", &result);
  if (status == napi_ok) {
    auto toString = Napi::Value(env, result).As<Napi::Function>();
    // NOT the same as .Call(). Need _this_ bound to the value or we throw.
    auto string = toString.Call(v, {});
    return string.As<Napi::String>().Utf8Value();
  } else {
    return "<no toString()>";
  }
}

void ThrowPythonException(Napi::Env env) {
  PyThreadStateLock py_thread_lock;

  Napi::HandleScope scope(env);

  PyObject* py_exception_type = NULL;
  PyObject* py_exception_value = NULL;
  PyObject* py_exception_traceback = NULL;

  PyErr_Fetch(&py_exception_type, &py_exception_value, &py_exception_traceback);

  // No exception from python
  if (py_exception_type == NULL)
    return;

  Napi::Error js_exception = ConvertToJSException(
      env, py_exception_type, py_exception_value, py_exception_traceback);

#if DEBUG
  // Print exception backtrace in Debug mode
  if (py_exception_traceback != NULL) {
    PyObject* py_exception_traceback_string =
        PyObject_Str(py_exception_traceback);
    printf("Throwing Py Exception as JS: %s\n",
           PyString_AsString(py_exception_traceback_string));
    Py_XDECREF(py_exception_traceback_string);
  }
#endif
  Py_XDECREF(py_exception_type);
  Py_XDECREF(py_exception_value);
  Py_XDECREF(py_exception_traceback);

  js_exception.ThrowAsJavaScriptException();
}

Napi::Error ConvertToJSException(Napi::Env env,
                                 PyObject* py_exception_type,
                                 PyObject* py_exception_value,
                                 PyObject* py_exception_traceback) {
  if (py_exception_type == NULL)
    return Napi::Error::New(env, "No exception found");

  std::string js_message;
  if (py_exception_value != NULL) {
    if (PyObject_TypeCheck(py_exception_value, &PyUnicode_Type)) {
      // If Exception(String), just take the string
      js_message = PyUnicode_AsUTF8(py_exception_value);
    } else if (PyObject_TypeCheck(py_exception_value, &PyTuple_Type) &&
               PyTuple_Size(py_exception_value) > 0) {
      js_message = PyUnicode_AsUTF8(PyTuple_GetItem(py_exception_value, 0));
    } else {
      PyObject* py_exception_value_string = PyObject_Str(py_exception_value);
      js_message = PyUnicode_AsUTF8(py_exception_value_string);
      Py_XDECREF(py_exception_value_string);
    }
  } else {
    js_message = "Unknown exception";
  }

  js_message = "Python: " + js_message;

  Napi::Error js_exception;
  if (PyErr_GivenExceptionMatches(py_exception_type, PyExc_IndexError) != 0) {
    js_exception = Napi::RangeError::New(env, js_message);
  } else if (PyErr_GivenExceptionMatches(py_exception_type,
                                         PyExc_ReferenceError) != 0) {
    // TODO: not easy to throw a ReferenceError in N-API at the moment
    js_exception = Napi::Error::New(env, "SyntaxError: " + js_message);
  } else if (PyErr_GivenExceptionMatches(py_exception_type,
                                         PyExc_SyntaxError) != 0) {
    // TODO: not easy to throw a SyntaxError in N-API at the moment
    js_exception = Napi::Error::New(env, "SyntaxError: " + js_message);
  } else if (PyErr_GivenExceptionMatches(py_exception_type, PyExc_TypeError) !=
             0) {
    js_exception = Napi::TypeError::New(env, js_message);
  } else {
    js_exception = Napi::Error::New(env, js_message);
  }

  return js_exception;
}

// Napi::Error ConvertToJSException(Napi::Env env, PyObject* py_exception) {
//   Napi::EscapableHandleScope scope(env);
//   auto js_exception =
//       ConvertToJSException(env, py_exception, py_exception, NULL);
//   return js_exception.Value();
// }

static void print_typechecks(PyObject* py_object) {
  std::cout << "done with printing type" << std::endl;
  std::cout << "pyNumber: " << PyNumber_Check(py_object) << std::endl;
  std::cout << "pyCallable: " << PyCallable_Check(py_object) << std::endl;
  std::cout << "pyUnicode: " << PyUnicode_Check(py_object) << std::endl;
  std::cout << "pyExceptionInstance: " << PyExceptionInstance_Check(py_object)
            << std::endl;
  std::cout << "pyDict: " << PyDict_Check(py_object) << std::endl;
  std::cout << "pyArray: " << &PyArray_Type << std::endl;
}

Napi::String StringToJS(const Napi::Env& env, PyObject* py_object) {
  std::cout << "converting string:" << std::endl;
  PyUnicode_READY(py_object);

  switch (PyUnicode_KIND(py_object)) {
    case PyUnicode_1BYTE_KIND:
      // std::cout << "converting 1-byte str" << std::endl;
      return Napi::String::New(env,
                               (const char*)PyUnicode_1BYTE_DATA(py_object));
    case PyUnicode_2BYTE_KIND:
      std::cout << "converting 2-byte str" << std::endl;
      return Napi::String::New(
          env, (const char16_t*)PyUnicode_2BYTE_DATA(py_object));
    case PyUnicode_4BYTE_KIND: {
      std::cout << "DEBUG: asked to convert 4-byte str" << std::endl;
      PyObject* as_utf16 = PyUnicode_AsUTF16String(py_object);
      return Napi::String::New(env, PyBytes_AS_STRING(as_utf16));
    }
  }
  Napi::Error::New(env, "Couldn't convert Python string to Node")
      .ThrowAsJavaScriptException();
  return Napi::String::New(env, "");
}

Napi::Value ConvertToJS(const Napi::Env& env, PyObject* py_object) {
  // const char* tp_name = py_object ? py_object->ob_type->tp_name : "None";
  // std::cout << "ConvertToJS " << py_object << " " << tp_name << std::endl;
  // The Python API is not const-correct
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwritable-strings"
  PyThreadStateLock py_thread_lock;

  // Lets us return things
  Napi::EscapableHandleScope scope(env);

  // Not really supposed to be passed nullptr, but handle it anyway
  if (py_object == nullptr) {
    std::cout << "ConvertToJS passed nullptr. Does this mean there is an error?"
              << std::endl;
    ThrowPythonException(env);
    return env.Undefined();
  }
#if DEBUG
  print_typechecks(py_object);
#endif

  if (py_object == Py_None) {
    std::cout << "Found Null object" << std::endl;
    return env.Null();
  } else if (PyFloat_Check(py_object) != 0) {
    auto value = PyFloat_AsDouble(py_object);
    return scope.Escape(Napi::Number::New(env, value));
  } else if (PyLong_Check(py_object) != 0) {
    auto value = PyLong_AsLong(py_object);
    return scope.Escape(Napi::Number::New(env, value));
  } else if (PyUnicode_Check(py_object) != 0) {
    auto str = StringToJS(env, py_object);
    return scope.Escape(str);
  } else if (PyCallable_Check(py_object) != 0) {  // Function
    Py_XINCREF(py_object);
    auto js_function = WrapCallable(env, py_object);
    // Local<FunctionTemplate> js_function_template =
    //     FunctionTemplate::New(env, Call, New(py_object));
    // Local<Function> js_function = js_function_template->GetFunction();
    // Napi::Value js_function_name = NamedGetter(py_object, "func_name");
    // if (!js_function_name.IsEmpty())
    //   js_function->SetName(js_function_name->ToString());
    return scope.Escape(js_function);
  } else if (PyExceptionInstance_Check(py_object)) {
    std::cout << "Converting python exception unimplemented." << std::endl;
    // TODO: unit test create python exception, but don't throw
    // auto js_exception = ConvertToJSException(env, py_object);
    // return scope.Escape(js_exception);
    return scope.Escape(env.Null());
  } else if (PyDict_Check(py_object)) {
    std::cout << "Converting python dict." << std::endl;
    int length = (int)PyMapping_Length(py_object);
    Napi::Object js_object = Napi::Object::New(env);
    PyObject* py_keys = PyMapping_Keys(py_object);
    PyObject* py_values = PyMapping_Values(py_object);
    for (int i = 0; i < length; i++) {
      PyObject* py_key = PySequence_GetItem(py_keys, i);
      PyObject* py_value = PySequence_GetItem(py_values, i);
      Napi::Value js_key = ConvertToJS(env, py_key);
      Napi::Value js_value = ConvertToJS(env, py_value);
      Py_XDECREF(py_key);
      Py_XDECREF(py_value);
      js_object.Set(js_key, js_value);
    }
    Py_XDECREF(py_keys);
    Py_XDECREF(py_values);
    return scope.Escape(js_object);
  } else if (PyList_Check(py_object) || PyTuple_Check(py_object)) {
    int length = (int)PySequence_Length(py_object);
    Napi::Array js_array = Napi::Array::New(env, length);
    for (int i = 0; i < length; i++) {
      PyObject* py_item = PySequence_GetItem(py_object, i);
      Napi::Value js_item = ConvertToJS(env, py_item);
      Py_XDECREF(py_item);
      js_array.Set(i, js_item);
    }
    return scope.Escape(js_array);
    // } else if (PyArray_Check(py_object)) {
    //   std::cout << "Found numpy array." << std::endl;
  } else {
    // Any other kind of object, wrap it in a proxy to return to JS
    std::cout << "Convert generic (" << py_object->ob_type->tp_name << ")"
              << std::endl;
    // PyObject* module_dict = PyModule_GetDict(py_object);
    // Wrap the module dict
    auto wrapped = PyObjectProxyHandler::WrapObject(env, py_object);
    // std::cout << "start converting module." << std::endl;
    // auto dict = ConvertToJS(env, module_dict);
    // std::cout << "end converting module." << std::endl;
    return scope.Escape(wrapped);
  }

  return scope.Escape(Napi::Value());
#pragma GCC diagnostic pop
}

PyObject* ConvertToPy(Napi::Value value) {
  // std::cout << "ConvertToPy: " << CoerceToString(value)
  //           << " type: " << value.Type() << std::endl;
  auto type = value.Type();
  switch (type) {
    case napi_undefined:
    case napi_null:
      Py_INCREF(Py_None);
      return Py_None;
    case napi_boolean:
      return PyBool_FromLong((long)value.As<Napi::Boolean>().Value());
    case napi_number:
      // TODO: can we check with V8 to see if this is actually an int?
      // Ints representable up to 2^31
      return PyFloat_FromDouble(value.As<Napi::Number>().DoubleValue());
    case napi_string: {
      auto strv = value.As<Napi::String>().Utf8Value();
      return PyUnicode_FromStringAndSize(strv.c_str(), strv.length());
    }
    case napi_symbol: {
      // Can't exactly round-trip this with JS at the moment
      // Just convert a symbol to a string for now
      Napi::Symbol symo = value.As<Napi::Symbol>();
      auto strv = describe_symbol(symo);
      return PyUnicode_FromStringAndSize(strv.c_str(), strv.length());
    }
    case napi_object: {
      // Is this an array
      if (value.IsArray()) {
        auto arr = value.As<Napi::Array>();
        size_t length = arr.Length();
        PyObject* list = PyList_New(length);
        for (size_t i = 0; i < length; i++) {
          auto v = ConvertToPy(arr[i]);
          // This macro “steals” a reference to item
          PyList_SET_ITEM(list, i, v);
        }
        return list;
      } else if (value.IsTypedArray()) {
        // TODO: typed arrays
        Py_INCREF(Py_None);
        return Py_None;
      } else if (PyObjectProxyHandler::IsProxyValue(value)) {
        auto py_object = PyObjectProxyHandler::GetProxyValue(value).Data();
        Py_INCREF(py_object);
        return py_object;
      } else if (value.IsExternal()) {
        std::cout << "Value is Object said to be external" << std::endl;
        Py_INCREF(Py_None);
        return Py_None;
      } else {
        // For generality, convert a generic Object {k: v} -> Dictionary()
        // But, if there is a prototype, warn & in future do more full
        // conversion
        auto ov = value.As<Napi::Array>();
        auto props = ov.GetPropertyNames();
        size_t length = props.Length();

        PyObject* dict = _PyDict_NewPresized(length);
        for (size_t i = 0; i < length; i++) {
          auto value = Napi::Object::Value(props[i]);
          if (value.IsString()) {
            auto strv = value.As<Napi::String>();
            auto item = ov[strv];
            auto py_item = ConvertToPy(item);
            // Doesn't steal reference
            PyDict_SetItemString(dict, strv.Utf8Value().c_str(), py_item);
            Py_DECREF(py_item);
          } else {
            Napi::Error::New(value.Env(),
                             "python-node-embed erro.r: Tried to convert dict "
                             "with non-string key.")
                .ThrowAsJavaScriptException();
          }
        }

        return dict;
      }
    }
    case napi_function: {
      std::cout << "Asked to convert JS function to Py" << std::endl;

      return nullptr;
    }
    case napi_external: {
      std::cout << "Value is external" << std::endl;
      auto external = value.As<Napi::External<PyObject>>();
      PyObject* py_object = external.Data();
      Py_INCREF(py_object);
      return py_object;
    }
    case napi_bigint:
      return nullptr;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

std::string describe_symbol(Napi::Symbol symbol) {
  std::string str = CoerceToString(symbol);
  if (str != "") {
    return str;
  } else {
    return "<Bad Symbol>";
  }
}

std::string describe_napi_value_type(Napi::Value value) {
  switch (value.Type()) {
    case napi_undefined:
      return "undefined";
    case napi_null:
      return "null";
    case napi_boolean:
      return "boolean";
    case napi_number:
      return "number";
    case napi_string:
      return "string";
    case napi_symbol:
      return "symbol";

    case napi_object: {
      // Is this an array
      if (value.IsArray()) {
        return "object:Array";
      } else if (value.IsTypedArray()) {
        return "object:TypedArray";
      } else if (value.IsExternal()) {
        return "object:External";
      } else {
        return "object";
      }
    }
    case napi_function:
      return "function";

    case napi_external:
      return "external";

    case napi_bigint:
      return "bigint";
  }
}
