mod indexed_blocks;
mod multiset;
mod all_reduce_counts;
mod bpe;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn rustbpe(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bpe::bpe, m)?)?;  
    Ok(())
}
