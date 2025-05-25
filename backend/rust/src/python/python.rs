use crate::{crop_image, resize_image, resize_image_bytes};
// #[cfg(feature = "python")]
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    pyfunction, pymodule,
    types::{PyBytes, PyModule},
    wrap_pyfunction, Python,
};

// #[cfg(feature = "python")]
#[pyfunction(name = "resize_image_bytes")]
pub fn resize_image_bytes_py(py: Python<'_>, data: &[u8], max_dim: u32) -> PyResult<PyObject> {
    let out =
        resize_image_bytes(data, max_dim).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &out).into())
}

// #[cfg(feature = "python")]
#[pyfunction(name = "resize_image")]
pub fn resize_image_py(_py: Python, inp: &str, outp: &str, max_dim: u32) -> PyResult<()> {
    resize_image(inp, outp, max_dim)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

// #[cfg(feature = "python")]
#[pyfunction(name = "crop_image")]
pub fn crop_image_py(
    py: Python<'_>,
    data: &[u8],
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> PyResult<PyObject> {
    let out = crop_image(&data, x, y, w, h).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &out).into())
}

// #[cfg(feature = "python")]
#[pymodule(name = "zendolib")]
pub fn zendolib_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resize_image_py, m)?)?;
    m.add_function(wrap_pyfunction!(resize_image_bytes_py, m)?)?;
    m.add_function(wrap_pyfunction!(crop_image_py, m)?)?;
    Ok(())
}
