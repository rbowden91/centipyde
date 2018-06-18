// taken mostly from http://www.codeproject.com/Articles/11805/Embedding-Python-in-C-C-Part-I
// TODO: check for errors

#include <Python.h>

int main(int argc, char *argv[])
{
    PyObject *pName, *pModule, *pDict, *pFunc;

    // Initialize the Python Interpreter
    Py_Initialize();

    PyObject *sys = PyImport_ImportModule("sys");
    PyObject *path = PyObject_GetAttrString(sys, "path");
    PyList_Append(path, PyUnicode_FromString("."));

    // Build the name object
    pName = PyUnicode_FromString("cwrapper");

    // Load the module object
    pModule = PyImport_Import(pName);

    // pDict is a borrowed reference
    pDict = PyModule_GetDict(pModule);

    // pFunc is also a borrowed reference
    pFunc = PyDict_GetItemString(pDict, "LibCWrapper");

    if (PyCallable_Check(pFunc))
    {
        PyObject_CallObject(pFunc, NULL);
    } else
    {
        PyErr_Print();
    }

    // Clean up
    Py_DECREF(pModule);
    Py_DECREF(pName);

    // Finish the Python Interpreter
    Py_Finalize();

    return 0;
}
