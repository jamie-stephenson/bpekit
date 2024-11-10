mod datastructures;
mod comms;

use crate::utils::progress::Progress;

use std::collections::HashMap;

use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyIterator,PyString};

#[pyfunction]
pub fn train(
    generator: &Bound<'_, PyIterator>, 
    vocab_size: u32, 
) -> PyResult<Vec<((u32, u32), u32)>> {
    
    // Convert PyIterator into iterator that yields `String`s
    let local_blocks = generator
        .into_iter()    
        .map(|s| {
            // TODO: Handle Result better, don't just unwrap
            s
            .unwrap()
            .downcast::<PyString>()
            .unwrap()
            .to_string()
        });

    let (blocks, mut bp_counts) = datastructures::init(local_blocks);
    
    let mut current_vocab_size: u32 = 256;
    let mut merges: Vec<((u32, u32), u32)> = Vec::with_capacity((vocab_size-current_vocab_size) as usize); 
    
    let mut progress = Progress::new(
        Some((vocab_size-current_vocab_size) as usize), //length
        0,
        "🤝 merging pairs",
        Some("🤝 pairs merged"),
    );
    
    while current_vocab_size < vocab_size {
        
        let pair = match bp_counts.pop() { 
            Some(item) => item,
            None => break // Exit the loop if no pairs are left
        };
        
        if bp_counts.is_stale(&pair) {
            bp_counts.update_count_and_push(pair);
            continue;
        }
        
        let (left,right) = pair.vals;

        merges.push((pair.vals, current_vocab_size));
        
        let changes: HashMap<(u32, u32), (i32, Vec<usize>)> = pair.block_ids
            .par_iter()
            .map(|&block_idx| {
                let block = &blocks[block_idx] as *const _ as *mut datastructures::Block;
                unsafe {
                    (*block).merge(left, right, current_vocab_size, block_idx)
                }
            })
            .reduce(
                || HashMap::new(),
                |mut changes1, changes2| {
                    for (pair, (change2, block_ids)) in changes2 {
                        changes1
                        .entry(pair)
                        .and_modify(|(change1,ids)| {
                            *change1 += change2;
                            ids.append(&mut block_ids.clone());
                        })                             
                        .or_insert((change2,block_ids));
                }
                changes1
            });

        // Commit changes and sync across ranks
        bp_counts.commit(changes);
        
        current_vocab_size += 1;
        
        progress.inc(1);
        //println!("New bytepair merge {:?} -> {:?} with count {:?}.", pair.vals, current_vocab_size, pair.count);
    }

    progress.finish();
    Ok(merges)
}