/* helion/_helion_c_launcher.c
 *
 * G6-launcher-C: C extension for the ``_DirectCallKernel`` locked hot
 * path inside Helion's Pallas/TPU launcher.  The pure-Python locked
 * path (``full_invoke_pure_output`` in
 * ``helion/runtime/__init__.py``) pays ~5-8 us per call to CPython
 * frame setup, list-comp dispatch, kwargs dict construction, and
 * three module-global counter bumps.  This extension folds the same
 * work into a single ``PyCFunction`` call that:
 *
 *   1. Increments three C-level counters (mirrored back to the Python
 *      counters lazily by ``_call_custom_kernel_direct_hits`` /
 *      ``_jaxcallable_key_cache_hits`` /
 *      ``_direct_call_sig_checks_skipped`` reads, which is the
 *      pin-test contract).
 *   2. Builds the ``input_tensors`` list by walking the pre-captured
 *      ``tensor_arg_indices`` tuple and calling ``.contiguous()`` on
 *      each ``args[i]``.
 *   3. Calls ``call_custom_kernel(kernel_name, kernel_key,
 *      inputs=input_tensors, output_shapes=output_shapes,
 *      donate_argnums=donate_argnums)`` via ``PyObject_Call`` with a
 *      pre-built kwargs dict (constructed once at context creation
 *      time, then reused per call by inserting a fresh
 *      ``input_tensors`` reference at the ``inputs`` slot).
 *   4. Calls ``out_tree.unflatten(results)``.
 *
 * Two ``DirectCallContext`` variants exist:
 *
 *   - ``DirectCallPureOutputContext`` — matches
 *     ``full_invoke_pure_output`` (output-only kernel, no aliases):
 *     after step 4 returns directly.
 *   - ``DirectCallInplaceContext`` — matches
 *     ``full_invoke_inplace_only`` (in-place kernel with aliases):
 *     after step 3, iterates ``alias_items`` calling
 *     ``input_tensors[in_idx].copy_(results[out_idx])`` then returns
 *     ``None``.
 *
 * Both contexts hold strong references to all captured constants so
 * the Python ``_DirectCallKernel`` only needs to keep the context
 * itself alive.
 *
 * If the extension fails to import (missing compiler, ABI mismatch,
 * etc.) the Python locked path in ``_build_direct_call_full_invoke``
 * stays — there is no behavioural regression.  The Python wrapper
 * checks for the extension at module load and exposes
 * ``_C_EXTENSION_AVAILABLE``.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* C-level mirrors of the three module-global counters.  Each locked
 * call bumps all three by 1; the Python-side counter getters lazily
 * pull these into the module globals so pin tests reading
 * ``helion.runtime._call_custom_kernel_direct_hits()`` etc. see the
 * combined Python + C count. */
static long _c_call_custom_kernel_direct_hits = 0;
static long _c_jaxcallable_key_cache_hits = 0;
static long _c_direct_call_sig_checks_skipped = 0;

/* Forward declarations */
typedef struct {
    PyObject_HEAD
    /* Pre-captured ``call_custom_kernel`` (torch_tpu function ref) */
    PyObject *call_custom_kernel;
    /* Pre-built positional args tuple (kernel_name, kernel_key) */
    PyObject *positional_args;
    /* Pre-built kwargs dict template with ``output_shapes`` and
     * ``donate_argnums`` already inserted.  Per-call we insert
     * ``inputs`` and call, then delete ``inputs`` so the dict is
     * ready for the next call.  Both inserts/deletes are O(1) on a
     * dict of size 3. */
    PyObject *kwargs_dict;
    /* Interned key strings to avoid per-call PyUnicode_InternFromString. */
    PyObject *inputs_key;
    /* Pre-captured ``out_tree.unflatten`` bound method (so the hot path
     * does not re-attribute-walk on each call). */
    PyObject *unflatten;
    /* Pre-captured ``tensor_arg_indices`` tuple, used for the
     * input-tensor walk.  Stored as a tuple of PyLong objects so the
     * hot path uses PyLong_AsSsize_t (fast) instead of PyNumber_AsSsize_t
     * (general). */
    PyObject *tensor_arg_indices;
    /* Interned ``"contiguous"`` string for PyObject_GetAttr (avoids
     * a per-call lookup in the interned strings dict). */
    PyObject *contiguous_str;
    /* Cached number of tensor_arg_indices for fast loop bound. */
    Py_ssize_t n_inputs;
} DirectCallPureOutputContext;

typedef struct {
    PyObject_HEAD
    PyObject *call_custom_kernel;
    PyObject *positional_args;
    PyObject *kwargs_dict;
    PyObject *inputs_key;
    /* Pre-captured ``out_tree.unflatten`` (still used so the closure
     * can return the unflattened result for completeness, even though
     * the in-place variant currently returns None.  Keeping it
     * preserves Python parity if upstream ever changes the
     * full_invoke_inplace_only contract.) */
    PyObject *unflatten;
    PyObject *tensor_arg_indices;
    PyObject *contiguous_str;
    Py_ssize_t n_inputs;
    /* Tuple of (in_idx, out_idx) pairs as (PyLong, PyLong) tuples. */
    PyObject *alias_items;
    /* Interned ``"copy_"`` for the alias copy-back loop. */
    PyObject *copy_str;
    Py_ssize_t n_aliases;
} DirectCallInplaceContext;

/* ---- DirectCallPureOutputContext methods ---- */

static void
_pure_output_dealloc(DirectCallPureOutputContext *self)
{
    Py_XDECREF(self->call_custom_kernel);
    Py_XDECREF(self->positional_args);
    Py_XDECREF(self->kwargs_dict);
    Py_XDECREF(self->inputs_key);
    Py_XDECREF(self->unflatten);
    Py_XDECREF(self->tensor_arg_indices);
    Py_XDECREF(self->contiguous_str);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* The hot path.  Equivalent to:
 *
 *     def __call__(args):
 *         bump_counters()
 *         input_tensors = [args[i].contiguous() for i in tensor_arg_indices]
 *         results = call_custom_kernel(kernel_name, kernel_key,
 *             inputs=input_tensors, output_shapes=output_shapes,
 *             donate_argnums=donate_argnums)
 *         return out_tree.unflatten(results)
 *
 * ``args`` arrives as a single positional argument (a tuple of the
 * full launcher arg list).
 */
static PyObject *
_pure_output_call(DirectCallPureOutputContext *self, PyObject *args, PyObject *kwds)
{
    /* Expect exactly one positional argument: the args tuple from the
     * launcher.  No kwargs supported (the Python closure also takes
     * positional-only). */
    PyObject *launcher_args = NULL;
    if (!PyArg_ParseTuple(args, "O:DirectCallPureOutput.__call__", &launcher_args)) {
        return NULL;
    }

    /* Bump the three C-level counters. */
    _c_call_custom_kernel_direct_hits++;
    _c_jaxcallable_key_cache_hits++;
    _c_direct_call_sig_checks_skipped++;

    /* Build input_tensors = [args[i].contiguous() for i in tensor_arg_indices]
     * using direct tuple access for speed. */
    PyObject *input_tensors = PyList_New(self->n_inputs);
    if (input_tensors == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self->n_inputs; i++) {
        /* tensor_arg_indices is a tuple of PyLongs; PyTuple_GET_ITEM
         * is borrowed, PyLong_AsSsize_t is fast for small longs. */
        PyObject *idx_obj = PyTuple_GET_ITEM(self->tensor_arg_indices, i);
        Py_ssize_t idx = PyLong_AsSsize_t(idx_obj);
        if (idx == -1 && PyErr_Occurred()) {
            Py_DECREF(input_tensors);
            return NULL;
        }
        /* launcher_args is a tuple from the launcher; use the safe
         * accessor (not _GET_ITEM) so we surface a clear error if it's
         * the wrong shape. */
        PyObject *arg = PyTuple_GetItem(launcher_args, idx);
        if (arg == NULL) {
            Py_DECREF(input_tensors);
            return NULL;
        }
        /* arg.contiguous() — use the interned "contiguous" attr name. */
        PyObject *contiguous_method = PyObject_GetAttr(arg, self->contiguous_str);
        if (contiguous_method == NULL) {
            Py_DECREF(input_tensors);
            return NULL;
        }
        PyObject *contig_result = PyObject_CallNoArgs(contiguous_method);
        Py_DECREF(contiguous_method);
        if (contig_result == NULL) {
            Py_DECREF(input_tensors);
            return NULL;
        }
        /* PyList_SET_ITEM steals the reference (and we own it). */
        PyList_SET_ITEM(input_tensors, i, contig_result);
    }

    /* Insert input_tensors into the pre-built kwargs dict.  After the
     * call we delete it so subsequent calls start from a 2-entry dict
     * again (output_shapes + donate_argnums). */
    if (PyDict_SetItem(self->kwargs_dict, self->inputs_key, input_tensors) < 0) {
        Py_DECREF(input_tensors);
        return NULL;
    }
    /* Drop our owning reference to input_tensors; the dict holds one. */
    Py_DECREF(input_tensors);

    /* Call ``call_custom_kernel(*positional_args, **kwargs_dict)``. */
    PyObject *results = PyObject_Call(
        self->call_custom_kernel, self->positional_args, self->kwargs_dict);

    /* Clear the inputs slot so the dict is ready for the next call.
     * Failure here is unlikely (the key was set above) but we still
     * tolerate it. */
    if (PyDict_DelItem(self->kwargs_dict, self->inputs_key) < 0) {
        /* Clear the exception; the call itself succeeded (or failed)
         * and that result takes precedence. */
        PyErr_Clear();
    }

    if (results == NULL) {
        return NULL;
    }

    /* out_tree.unflatten(results) — use a single-arg fast-call shape. */
    PyObject *unflatten_args[1] = {results};
    PyObject *out = PyObject_Vectorcall(self->unflatten, unflatten_args, 1, NULL);
    Py_DECREF(results);
    return out;
}

static PyTypeObject DirectCallPureOutputType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_helion_c_launcher.DirectCallPureOutput",
    .tp_basicsize = sizeof(DirectCallPureOutputContext),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)_pure_output_dealloc,
    .tp_call = (ternaryfunc)_pure_output_call,
    /* No GC support — captured references are all to immutable
     * Python objects (torch_tpu function, str, tuple, dict, bound
     * method); no cycles possible. */
};

/* ---- DirectCallInplaceContext methods ---- */

static void
_inplace_dealloc(DirectCallInplaceContext *self)
{
    Py_XDECREF(self->call_custom_kernel);
    Py_XDECREF(self->positional_args);
    Py_XDECREF(self->kwargs_dict);
    Py_XDECREF(self->inputs_key);
    Py_XDECREF(self->unflatten);
    Py_XDECREF(self->tensor_arg_indices);
    Py_XDECREF(self->contiguous_str);
    Py_XDECREF(self->alias_items);
    Py_XDECREF(self->copy_str);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
_inplace_call(DirectCallInplaceContext *self, PyObject *args, PyObject *kwds)
{
    PyObject *launcher_args = NULL;
    if (!PyArg_ParseTuple(args, "O:DirectCallInplace.__call__", &launcher_args)) {
        return NULL;
    }

    _c_call_custom_kernel_direct_hits++;
    _c_jaxcallable_key_cache_hits++;
    _c_direct_call_sig_checks_skipped++;

    PyObject *input_tensors = PyList_New(self->n_inputs);
    if (input_tensors == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self->n_inputs; i++) {
        PyObject *idx_obj = PyTuple_GET_ITEM(self->tensor_arg_indices, i);
        Py_ssize_t idx = PyLong_AsSsize_t(idx_obj);
        if (idx == -1 && PyErr_Occurred()) {
            Py_DECREF(input_tensors);
            return NULL;
        }
        PyObject *arg = PyTuple_GetItem(launcher_args, idx);
        if (arg == NULL) {
            Py_DECREF(input_tensors);
            return NULL;
        }
        PyObject *contiguous_method = PyObject_GetAttr(arg, self->contiguous_str);
        if (contiguous_method == NULL) {
            Py_DECREF(input_tensors);
            return NULL;
        }
        PyObject *contig_result = PyObject_CallNoArgs(contiguous_method);
        Py_DECREF(contiguous_method);
        if (contig_result == NULL) {
            Py_DECREF(input_tensors);
            return NULL;
        }
        PyList_SET_ITEM(input_tensors, i, contig_result);
    }

    if (PyDict_SetItem(self->kwargs_dict, self->inputs_key, input_tensors) < 0) {
        Py_DECREF(input_tensors);
        return NULL;
    }

    PyObject *results = PyObject_Call(
        self->call_custom_kernel, self->positional_args, self->kwargs_dict);

    if (PyDict_DelItem(self->kwargs_dict, self->inputs_key) < 0) {
        PyErr_Clear();
    }

    if (results == NULL) {
        Py_DECREF(input_tensors);
        return NULL;
    }

    /* For each (in_idx, out_idx) in alias_items:
     *     input_tensors[in_idx].copy_(results[out_idx]) */
    for (Py_ssize_t i = 0; i < self->n_aliases; i++) {
        PyObject *pair = PyTuple_GET_ITEM(self->alias_items, i);
        PyObject *in_idx_obj = PyTuple_GET_ITEM(pair, 0);
        PyObject *out_idx_obj = PyTuple_GET_ITEM(pair, 1);
        Py_ssize_t in_idx = PyLong_AsSsize_t(in_idx_obj);
        Py_ssize_t out_idx = PyLong_AsSsize_t(out_idx_obj);
        if ((in_idx == -1 || out_idx == -1) && PyErr_Occurred()) {
            Py_DECREF(input_tensors);
            Py_DECREF(results);
            return NULL;
        }
        PyObject *target = PyList_GET_ITEM(input_tensors, in_idx);
        PyObject *source = PySequence_GetItem(results, out_idx);
        if (source == NULL) {
            Py_DECREF(input_tensors);
            Py_DECREF(results);
            return NULL;
        }
        PyObject *copy_method = PyObject_GetAttr(target, self->copy_str);
        if (copy_method == NULL) {
            Py_DECREF(source);
            Py_DECREF(input_tensors);
            Py_DECREF(results);
            return NULL;
        }
        PyObject *copy_args[1] = {source};
        PyObject *copy_result = PyObject_Vectorcall(copy_method, copy_args, 1, NULL);
        Py_DECREF(copy_method);
        Py_DECREF(source);
        if (copy_result == NULL) {
            Py_DECREF(input_tensors);
            Py_DECREF(results);
            return NULL;
        }
        Py_DECREF(copy_result);
    }

    Py_DECREF(input_tensors);
    Py_DECREF(results);
    Py_RETURN_NONE;
}

static PyTypeObject DirectCallInplaceType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_helion_c_launcher.DirectCallInplace",
    .tp_basicsize = sizeof(DirectCallInplaceContext),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)_inplace_dealloc,
    .tp_call = (ternaryfunc)_inplace_call,
};

/* ---- Factory functions ---- */

/* build_pure_output_context(call_custom_kernel, kernel_name,
 *                           kernel_key, output_shapes, donate_argnums,
 *                           out_tree, tensor_arg_indices) */
static PyObject *
build_pure_output_context(PyObject *module, PyObject *args)
{
    PyObject *call_custom_kernel, *kernel_name, *kernel_key;
    PyObject *output_shapes, *donate_argnums, *out_tree;
    PyObject *tensor_arg_indices;
    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &call_custom_kernel, &kernel_name, &kernel_key,
                          &output_shapes, &donate_argnums, &out_tree,
                          &tensor_arg_indices)) {
        return NULL;
    }
    if (!PyTuple_Check(tensor_arg_indices)) {
        PyErr_SetString(PyExc_TypeError,
                        "tensor_arg_indices must be a tuple");
        return NULL;
    }

    DirectCallPureOutputContext *ctx = PyObject_New(
        DirectCallPureOutputContext, &DirectCallPureOutputType);
    if (ctx == NULL) {
        return NULL;
    }
    /* Zero-init optional pointers so dealloc on a failed init below
     * doesn't double-free. */
    ctx->call_custom_kernel = NULL;
    ctx->positional_args = NULL;
    ctx->kwargs_dict = NULL;
    ctx->inputs_key = NULL;
    ctx->unflatten = NULL;
    ctx->tensor_arg_indices = NULL;
    ctx->contiguous_str = NULL;
    ctx->n_inputs = 0;

    Py_INCREF(call_custom_kernel);
    ctx->call_custom_kernel = call_custom_kernel;

    /* Pre-build the (kernel_name, kernel_key) positional tuple. */
    ctx->positional_args = PyTuple_Pack(2, kernel_name, kernel_key);
    if (ctx->positional_args == NULL) {
        Py_DECREF(ctx);
        return NULL;
    }

    /* Pre-build the kwargs dict with output_shapes + donate_argnums
     * already inserted. */
    ctx->kwargs_dict = PyDict_New();
    if (ctx->kwargs_dict == NULL) {
        Py_DECREF(ctx);
        return NULL;
    }
    if (PyDict_SetItemString(ctx->kwargs_dict, "output_shapes", output_shapes) < 0 ||
        PyDict_SetItemString(ctx->kwargs_dict, "donate_argnums", donate_argnums) < 0) {
        Py_DECREF(ctx);
        return NULL;
    }

    ctx->inputs_key = PyUnicode_InternFromString("inputs");
    if (ctx->inputs_key == NULL) {
        Py_DECREF(ctx);
        return NULL;
    }

    /* Grab the bound out_tree.unflatten method once. */
    ctx->unflatten = PyObject_GetAttrString(out_tree, "unflatten");
    if (ctx->unflatten == NULL) {
        Py_DECREF(ctx);
        return NULL;
    }

    Py_INCREF(tensor_arg_indices);
    ctx->tensor_arg_indices = tensor_arg_indices;
    ctx->n_inputs = PyTuple_GET_SIZE(tensor_arg_indices);

    ctx->contiguous_str = PyUnicode_InternFromString("contiguous");
    if (ctx->contiguous_str == NULL) {
        Py_DECREF(ctx);
        return NULL;
    }

    return (PyObject *)ctx;
}

/* build_inplace_context(call_custom_kernel, kernel_name, kernel_key,
 *                       output_shapes, donate_argnums, out_tree,
 *                       tensor_arg_indices, alias_items) */
static PyObject *
build_inplace_context(PyObject *module, PyObject *args)
{
    PyObject *call_custom_kernel, *kernel_name, *kernel_key;
    PyObject *output_shapes, *donate_argnums, *out_tree;
    PyObject *tensor_arg_indices, *alias_items;
    if (!PyArg_ParseTuple(args, "OOOOOOOO",
                          &call_custom_kernel, &kernel_name, &kernel_key,
                          &output_shapes, &donate_argnums, &out_tree,
                          &tensor_arg_indices, &alias_items)) {
        return NULL;
    }
    if (!PyTuple_Check(tensor_arg_indices)) {
        PyErr_SetString(PyExc_TypeError,
                        "tensor_arg_indices must be a tuple");
        return NULL;
    }
    if (!PyTuple_Check(alias_items)) {
        PyErr_SetString(PyExc_TypeError, "alias_items must be a tuple");
        return NULL;
    }

    DirectCallInplaceContext *ctx = PyObject_New(
        DirectCallInplaceContext, &DirectCallInplaceType);
    if (ctx == NULL) {
        return NULL;
    }
    ctx->call_custom_kernel = NULL;
    ctx->positional_args = NULL;
    ctx->kwargs_dict = NULL;
    ctx->inputs_key = NULL;
    ctx->unflatten = NULL;
    ctx->tensor_arg_indices = NULL;
    ctx->contiguous_str = NULL;
    ctx->alias_items = NULL;
    ctx->copy_str = NULL;
    ctx->n_inputs = 0;
    ctx->n_aliases = 0;

    Py_INCREF(call_custom_kernel);
    ctx->call_custom_kernel = call_custom_kernel;

    ctx->positional_args = PyTuple_Pack(2, kernel_name, kernel_key);
    if (ctx->positional_args == NULL) {
        Py_DECREF(ctx);
        return NULL;
    }
    ctx->kwargs_dict = PyDict_New();
    if (ctx->kwargs_dict == NULL) {
        Py_DECREF(ctx);
        return NULL;
    }
    if (PyDict_SetItemString(ctx->kwargs_dict, "output_shapes", output_shapes) < 0 ||
        PyDict_SetItemString(ctx->kwargs_dict, "donate_argnums", donate_argnums) < 0) {
        Py_DECREF(ctx);
        return NULL;
    }
    ctx->inputs_key = PyUnicode_InternFromString("inputs");
    if (ctx->inputs_key == NULL) {
        Py_DECREF(ctx);
        return NULL;
    }
    ctx->unflatten = PyObject_GetAttrString(out_tree, "unflatten");
    if (ctx->unflatten == NULL) {
        Py_DECREF(ctx);
        return NULL;
    }
    Py_INCREF(tensor_arg_indices);
    ctx->tensor_arg_indices = tensor_arg_indices;
    ctx->n_inputs = PyTuple_GET_SIZE(tensor_arg_indices);
    ctx->contiguous_str = PyUnicode_InternFromString("contiguous");
    if (ctx->contiguous_str == NULL) {
        Py_DECREF(ctx);
        return NULL;
    }
    Py_INCREF(alias_items);
    ctx->alias_items = alias_items;
    ctx->n_aliases = PyTuple_GET_SIZE(alias_items);
    ctx->copy_str = PyUnicode_InternFromString("copy_");
    if (ctx->copy_str == NULL) {
        Py_DECREF(ctx);
        return NULL;
    }

    return (PyObject *)ctx;
}

/* Counter accessors / resetters. */
static PyObject *
get_counters(PyObject *module, PyObject *Py_UNUSED(args))
{
    return Py_BuildValue("(lll)",
                         _c_call_custom_kernel_direct_hits,
                         _c_jaxcallable_key_cache_hits,
                         _c_direct_call_sig_checks_skipped);
}

static PyObject *
reset_counters(PyObject *module, PyObject *Py_UNUSED(args))
{
    _c_call_custom_kernel_direct_hits = 0;
    _c_jaxcallable_key_cache_hits = 0;
    _c_direct_call_sig_checks_skipped = 0;
    Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
    {"build_pure_output_context", build_pure_output_context, METH_VARARGS,
     "Build a DirectCallPureOutput context for the locked hot path."},
    {"build_inplace_context", build_inplace_context, METH_VARARGS,
     "Build a DirectCallInplace context for the locked hot path."},
    {"get_counters", get_counters, METH_NOARGS,
     "Return (call_custom_kernel_direct_hits, jaxcallable_key_cache_hits, "
     "direct_call_sig_checks_skipped) tracked by the C extension."},
    {"reset_counters", reset_counters, METH_NOARGS,
     "Reset the C-side counters to zero."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "_helion_c_launcher",
    "Helion C extension for the Pallas direct-call locked hot path.",
    -1,
    ModuleMethods,
};

PyMODINIT_FUNC
PyInit__helion_c_launcher(void)
{
    if (PyType_Ready(&DirectCallPureOutputType) < 0) {
        return NULL;
    }
    if (PyType_Ready(&DirectCallInplaceType) < 0) {
        return NULL;
    }
    PyObject *m = PyModule_Create(&ModuleDef);
    if (m == NULL) {
        return NULL;
    }
    Py_INCREF(&DirectCallPureOutputType);
    if (PyModule_AddObject(m, "DirectCallPureOutput",
                           (PyObject *)&DirectCallPureOutputType) < 0) {
        Py_DECREF(&DirectCallPureOutputType);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&DirectCallInplaceType);
    if (PyModule_AddObject(m, "DirectCallInplace",
                           (PyObject *)&DirectCallInplaceType) < 0) {
        Py_DECREF(&DirectCallInplaceType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
