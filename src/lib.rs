pub mod train;
pub mod encode;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn rustbpe(_py: Python, m: &Bound<'_,PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train::train, m)?)?;  
    m.add_function(wrap_pyfunction!(encode::encode, m)?)?;  
    Ok(())
}