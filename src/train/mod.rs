mod datastructures;
mod comms;

use datastructures::{PairCounter,Block};
use crate::utils::progress::get_progress_reporter;

use std::collections::HashMap;
use std::time::Instant;

use counter::Counter;
use rayon::prelude::*;
use mpi::initialize;
use mpi::topology::Communicator;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyString};

#[pyfunction]
pub fn train(
    generator: &Bound<'_, PyIterator>, 
    vocab_size: u32, 
) -> PyResult<HashMap<(u32, u32), u32>> {

    // Init comms
    let universe = initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    
    // Log times
    let start = Instant::now();

    if rank == 0 {
        println!("Running BPE training algorithm...");
    }

    // Count the duplicate blocks in the generator
    let mut block_count_progress = get_progress_reporter(
        None,
        rank,
        10000000,
        true
    );
    
    let block_counts: Counter<String> = generator
        .into_iter()
        .map(|s| {
            block_count_progress.inc(1);
            // TODO: Error handling instead of unwrapping 
            s.unwrap()
                .downcast::<PyString>().unwrap()
                .to_str().unwrap()
                .to_string()
        })
        .collect();

    // Convert block_counts to blocks
    let blocks: Vec<Block> = block_counts
        .into_iter()
        .par_bridge()
        .map(|(s, count)| Block::new(s, count as i32))
        .collect();

    // Init pair counter
    let mut bp_counts = PairCounter::new(&blocks,&world);    

    // Log times
    if rank == 0 {
        let init_duration = start.elapsed();
        println!("Initialising datastuctures took {:.2} seconds", init_duration.as_secs());
    }

    // Begin training
    let mut current_vocab_size: u32 = 256;
    let mut merges: HashMap<(u32, u32), u32> = HashMap::new(); 

    let mut progress = get_progress_reporter(
        Some((vocab_size-current_vocab_size) as u64),
        rank,
        1000,
        true
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

        merges.insert(pair.vals, current_vocab_size);
        
        let changes: Vec<((u32, u32), (i32, Vec<usize>))> = pair.block_ids
            .par_iter()
            .map(|&block_idx| {
                let block = &blocks[block_idx] as *const _ as *mut Block;
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
                },
            ).into_iter().collect();

            // Commit changes and sync across ranks
            bp_counts.commit(changes);
            
            current_vocab_size += 1;

            progress.inc(1);
            //println!("New bytepair merge {:?} -> {:?} with count {:?}.", pair.vals, current_vocab_size, pair.count);
    }

    progress.finish();
    
    Ok(merges)
}