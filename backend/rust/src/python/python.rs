use crate::{crop_image, infer::ClipModel, resize_image, resize_image_bytes, transform_image};

use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyBytes};

use image::DynamicImage;
use numpy::{
    array::PyArray1, IntoPyArray, IxDyn, PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use std::convert::TryFrom;
use tch::{Device, Kind, Tensor};

#[pyfunction(name = "resize_image_bytes")]
pub fn resize_image_bytes_py(py: Python<'_>, data: &[u8], max_dim: u32) -> PyResult<PyObject> {
    let out =
        resize_image_bytes(data, max_dim).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &out).into())
}

#[pyfunction(name = "resize_image")]
pub fn resize_image_py(_py: Python, inp: &str, outp: &str, max_dim: u32) -> PyResult<()> {
    resize_image(inp, outp, max_dim)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

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

#[pyfunction(name = "transform_image")]
pub fn transform_image_py(py: Python<'_>, path: &str, scale: f32) -> PyResult<PyObject> {
    let img: DynamicImage =
        transform_image(path, scale).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let rgb = img.to_rgb8();
    let raw = rgb.into_raw();

    Ok(PyBytes::new(py, &raw).into())
}

#[pyclass]
pub struct PyClipModel {
    model: ClipModel,
}

#[pymethods]
impl PyClipModel {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        Ok(Self {
            model: ClipModel::new(&path)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create model: {e}")))?,
        })
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<PyObject> {
        let shape: Vec<i64> = input.shape().iter().map(|&n| n as i64).collect();

        let data = input
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let tensor = Tensor::from_slice(data)
            .reshape(&shape)
            .to_device(Device::Cpu);

        let output = tch::no_grad(|| {
            self.model
                .encode_image(&tensor)
                .map_err(|e| PyRuntimeError::new_err(format!("Inference failed: {e}")))
        })?;

        output
            // .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .contiguous();

        let output = output.squeeze_dim(0);

        let numel = output.numel();
        let mut buf = vec![0f32; numel];
        output
            .f_copy_data(&mut buf, numel)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let values: Vec<f32> = buf.to_vec() as Vec<f32>;
        // .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let out_shape: Vec<usize> = output.size().iter().map(|&d| d as usize).collect();

        // let py_arr = PyArray::<f32, _>::from_vec(py, values);
        // let dyn_arr = py_arr
        //     .reshape(IxDyn(&out_shape))
        //     .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let array = PyArray::from_vec(py, values);

        Ok(array.as_any().clone().unbind())

        // Ok(dyn_arr.as_any().clone().unbind())
    }
}

#[pymodule(name = "zendolib")]
pub fn zendolib_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resize_image_py, m)?)?;
    m.add_function(wrap_pyfunction!(resize_image_bytes_py, m)?)?;
    m.add_function(wrap_pyfunction!(crop_image_py, m)?)?;
    m.add_function(wrap_pyfunction!(transform_image_py, m)?)?;
    // m.add_function(wrap_pyfunction!(predict_clip, m)?)?;
    let _ = m.add_class::<PyClipModel>();
    Ok(())
}
