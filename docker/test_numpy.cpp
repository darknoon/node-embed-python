#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>

// This file works on linux (import python & numpy, when built as executable)

void *do_numpy_import(void) {
  // This is a macro, so this function isn't useless
  // It actually returns an int from inside the macro
  import_array();
  // Silences warning
  return nullptr;
}

void test_numpy() {
  int dims[] = {2,3,1};
  int count = sizeof(dims) / sizeof(dims[0]);
  int typenum = 0;
  void *data = nullptr;
  PyArray_SimpleNew(count, dims, typenum, data);
}

int main(int argc, char *argv[]) {
  std::cout << "Python initializing..." << std::endl;
  Py_Initialize();
  // Test
  std::cout << "Importing numpy..." << std::endl;
  do_numpy_import();

  
  std::cout << "Done." << std::endl;
}