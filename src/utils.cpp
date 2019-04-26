#include "utils.h"

#include <iostream>
#include "py_object_wrapper.h"

// Silence some warnings. Don't need deprecated API.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

int do_numpy_import(void) {
  import_array();
  return 0;
}

Napi::Error ConvertToJSException(const Napi::Env& env,
                                 PyObject* py_exception_type,
                                 PyObject* py_exception_value,
                                 PyObject* py_exception_traceback) {
#if 0

  if (py_exception_type == NULL)
    return scope.Escape(
        Napi::Error(String::NewFromUtf8(env, "No exception found")));

  Local<String> js_message;
  if (py_exception_value != NULL) {
    if (PyObject_TypeCheck(py_exception_value, &PyUnicode_Type)) {
      js_message =
          String::NewFromUtf8(env, PyUnicode_AsUTF8(py_exception_value));
    } else if (PyObject_TypeCheck(py_exception_value, &PyTuple_Type) &&
               PyTuple_Size(py_exception_value) > 0) {
      js_message = String::NewFromUtf8(
          env, PyUnicode_AsUTF8(PyTuple_GetItem(py_exception_value, 0)));
    } else {
      PyObject* py_exception_value_string = PyObject_Str(py_exception_value);
      js_message =
          String::NewFromUtf8(env, PyUnicode_AsUTF8(py_exception_value_string));
      Py_XDECREF(py_exception_value_string);
    }
  } else {
    js_message = String::NewFromUtf8(env, "Unknown exception");
  }

  Local<Value> js_exception;
  if (PyErr_GivenExceptionMatches(py_exception_type, PyExc_IndexError) != 0) {
    js_exception = Exception::RangeError(js_message);
  } else if (PyErr_GivenExceptionMatches(py_exception_type,
                                         PyExc_ReferenceError) != 0) {
    js_exception = Exception::ReferenceError(js_message);
  } else if (PyErr_GivenExceptionMatches(py_exception_type,
                                         PyExc_SyntaxError) != 0) {
    js_exception = Exception::SyntaxError(js_message);
  } else if (PyErr_GivenExceptionMatches(py_exception_type, PyExc_TypeError) !=
             0) {
    js_exception = Exception::TypeError(js_message);
  } else {
    js_exception = Exception::Error(js_message);
  }

#endif
  return Napi::Error::New(env, "TODO: Convert this Python exception");
}

Napi::Value ConvertToJSException(const Napi::Env& env, PyObject* py_exception) {
  Napi::EscapableHandleScope scope(env);
  auto js_exception =
      ConvertToJSException(env, py_exception, py_exception, NULL);
  return scope.Escape(js_exception.Value());
}

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
  // The Python API is not const-correct
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwritable-strings"
  // PyThreadStateLock py_thread_lock;

  // Lets us return things
  Napi::EscapableHandleScope scope(env);

  // Not really supposed to be passed nullptr, but handle it anyway
  if (py_object == nullptr) {
    return scope.Escape(scope.Env().Null());
  }
#if DEBUG
  print_typechecks(py_object);
#endif

  if (PyNumber_Check(py_object) != 0) {
    auto value = PyFloat_AsDouble(py_object);
    return scope.Escape(Napi::Number::New(env, value));
  } else if (PyUnicode_Check(py_object) != 0) {
    auto str = StringToJS(env, py_object);
    return scope.Escape(str);
  } else if (PyCallable_Check(py_object) != 0) {  // Function
    Py_XINCREF(py_object);
    auto js_function = PyObjectProxyHandler::WrapCallable(env, py_object);
    // Local<FunctionTemplate> js_function_template =
    //     FunctionTemplate::New(env, Call, New(py_object));
    // Local<Function> js_function = js_function_template->GetFunction();
    // Napi::Value js_function_name = NamedGetter(py_object, "func_name");
    // if (!js_function_name.IsEmpty())
    //   js_function->SetName(js_function_name->ToString());
    return scope.Escape(js_function);
  } else if (PyExceptionInstance_Check(py_object)) {
    std::cout << "Converting python exception." << std::endl;
    auto js_exception = ConvertToJSException(env, py_object);
    return scope.Escape(js_exception);
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
  } else if (PyArray_Check(py_object)) {
    std::cout << "Found numpy array." << std::endl;

  } else {
    std::cout << "Converting python unknown object." << std::endl;
    PyObject* str_repr = PyObject_Str(py_object);

    // debug only
    if (str_repr != nullptr) {
      std::cout << "found string repr" << std::endl;
      PyUnicode_READY(str_repr);
      std::cout << "str: ." << PyUnicode_2BYTE_DATA(str_repr) << std::endl;
    }
    return scope.Escape(Napi::Value());
    // TODO: return the str rep instead
    // Py_XINCREF(py_object);
    // PyObject* py_object_repr = PyObject_Repr(py_object);
    // if (py_object_repr != py_object) {
    //   Napi::Value js_object = ConvertToJS(env, py_object_repr);
    //   return scope.Escape(js_object);
    // } else {
    // }
  }

  return Napi::Value();
#pragma GCC diagnostic pop
}

PyObject* ConvertToPy(Napi::Value value) {
  std::cout << "Converting value to Py: " << value.ToString().Utf8Value()
            << std::endl;
  auto type = value.Type();
  switch (type) {
    case napi_undefined:
    case napi_null:
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
      Napi::Symbol symo = value.As<Napi::Symbol>();
      auto symd = symo.ToString();
      auto strv = symd.Utf8Value();
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
        return nullptr;
      } else {
        // TODO: typed arrays
        // For generality, convert a generic Object {k: v} -> Dictionary()
        // But, if there is a prototype, warn & in future do more full
        // conversion
        auto ov = value.As<Napi::Array>();
        auto props = ov.GetPropertyNames();
        size_t length = props.Length();

        PyObject* dict = _PyDict_NewPresized(length);
        for (size_t i = 0; i < length; i++) {
          auto strv = Napi::Object::Value(props[i]).As<Napi::String>();
          auto item = ov[strv];
          auto py_item = ConvertToPy(item);
          // Doesn't steal reference
          PyDict_SetItemString(dict, strv.Utf8Value().c_str(), py_item);
          Py_DECREF(py_item);
        }

        return dict;
      }
    }
    case napi_function:
      return nullptr;
    case napi_external:
      return nullptr;
    case napi_bigint:
      return nullptr;
  }

  return PyFloat_FromDouble(0);
}