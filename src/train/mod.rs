mod datastructures;
mod iterators;

use datastructures::{PairCounter,Block};
use iterators::PyBufferedIterator;

use std::collections::HashMap;
use std::time::Instant;

use regex::Regex;
use rayon::prelude::*;
use mpi::initialize;
use mpi::topology::Communicator;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::exceptions::PyValueError;
use itertools::Either;

#[pyfunction]
pub fn train(generator: &Bound<'_, PyAny>, vocab_size: u32, pattern: &str) -> PyResult<HashMap<(u32, u32), u32>> {

    // Init comms
    let universe = initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    
    // Log times
    let start = Instant::now();

    if rank == 0 {
        println!("Running BPE training algorithm...");
    }

    // Covert Python generator into special iterator that handles Python GIL.
    // This allows us to access the generator from multiple threads without
    // upsetting the GIL. 
    let buffered_iter = match PyBufferedIterator::new(
        generator,
        |element| {
            match element.downcast::<PyString>() {
                Ok(s) => Either::Right(std::iter::once(s.to_str())),
                Err(_) => Either::Left(std::iter::once(Ok("_")))
            }
        },
        256,
    )
    {
        Ok(iter) => iter,
        Err(e) => panic!["Failed to convert python generator to rust `PyBufferedIterator`: {:?}",e]
    };

    // Split each doc into blocks using regex and then count duplicates of each block.
    let re = Regex::new(pattern).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let block_counts = buffered_iter
        .par_bridge()
        .map(|doc| {
            let strs: Vec<&str> = re.find_iter(doc.unwrap()).map(|mat| mat.as_str()).collect();
            let mut map: HashMap<&str, i32> = HashMap::new();
            for s in strs {
                map.entry(s).and_modify(|c| *c += 1).or_insert(1);
            }
            map
        })
        .reduce(
            || HashMap::new(),
            |mut a, b| {
                for (s, count) in b {
                    a.entry(s).and_modify(|c| *c += count).or_insert(count);
                }
                a
            },
        );


    // Convert block_counts to blocks
    let mut blocks: Vec<Block> = block_counts
        .into_par_iter()
        .map(|(s, count)| Block::new(s, count))
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

        if rank == 0 {
            println!("New bytepair merge {:?} -> {:?} with count {:?}.", pair.vals, current_vocab_size, pair.count);
        }

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

        bp_counts.commit(changes);

        current_vocab_size += 1;
    }

    if rank == 0 {
        let duration = start.elapsed();
        println!("BPE training algorithm complete in {:.2} seconds", duration.as_secs());
    }
    
    Ok(merges)
}