mod commands; 
mod utils;

use commands::{train,encode};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn rust(_py: Python, m: &Bound<'_,PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train::train, m)?)?;  
    m.add_function(wrap_pyfunction!(encode::encode, m)?)?;  
    m.add_function(wrap_pyfunction!(encode::encode_dataset, m)?)?;  
    Ok(())
}