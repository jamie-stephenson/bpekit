mod datastructures;
mod comms;
mod iter;

use datastructures::{PairCounter,Block};
use iter::PyBufferedIterator;
use crate::utils::get_progress_reporter;
use crate::utils::progress::ProgressReporter;

use std::collections::HashMap;

use indicatif::{ProgressIterator,ProgressBar};
use dashmap::DashMap;
use itertools::Either;
use rayon::prelude::*;
use mpi::initialize;
use mpi::topology::Communicator;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyString};

#[pyfunction]
pub fn train(
    py: Python,
    generator: &Bound<'_, PyIterator>, 
    vocab_size: u32, 
) -> PyResult<HashMap<(u32, u32), u32>> {

    // Init comms
    let universe = initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    
    if rank == 0 {
        println!("Running BPE training algorithm...");
    }

    // Count the duplicate blocks in the generator
    let block_count_progress = get_progress_reporter(
        None,      // length
        rank,
        "🧱 counting blocks",
        Some("🧱 blocks counted"),
        10000000,  // interval
        true       // verbose 
    );

    let spinner = match block_count_progress {
        ProgressReporter::Spinner(ps) => ps,
        _ => ProgressBar::new_spinner() 
    }; 

    // Covert Python generator into special iterator that handles Python GIL.
    // This allows us to access the generator from multiple threads without
    // upsetting the GIL. 
    let buffered_iter = match PyBufferedIterator::new(
        generator,
        |element| {
            match element.downcast::<PyString>() {
                Ok(s) => Either::Right(std::iter::once(s.to_str().map(|s| s.to_string()))),
                Err(_) => Either::Left(std::iter::once(Ok("_".to_string())))
            }
        },
        8132,
    )
    {
        Ok(iter) => iter,
        Err(e) => panic!["Failed to convert python generator to rust `PyBufferedIterator`: {:?}",e]
    };

    let block_counts: DashMap<String, i32> = DashMap::new();
    
    py.allow_threads(||{
        buffered_iter
            .into_iter()
            .progress_with(spinner)
            .par_bridge()
            .for_each(|s| {
                // TODO: Error handling instead of unwrapping 
                block_counts.entry(s.unwrap()).and_modify(|count| { *count += 1 }).or_insert(1);
            });
    });

    // Convert block_counts to blocks
    let mut block_tokenize_progress = get_progress_reporter(
        Some(block_counts.len()), // length
        rank,
        "🔢 tokenizing blocks",
        None,
        10000000,  // interval
        true       // verbose 
    );
    
    let blocks: Vec<Block> = block_counts
        .into_iter()
        .map(|(s, count)| {
            block_tokenize_progress.inc(1);
            Block::new(s, count as i32)
        })
        .collect();

    block_tokenize_progress.finish_with_message("🔢 blocks tokenized");

    // Init pair counter
    let mut bp_counts = PairCounter::new(&blocks,&world);    

    // Begin training
    let mut current_vocab_size: u32 = 256;
    let mut merges: HashMap<(u32, u32), u32> = HashMap::new(); 
    
    let mut progress = get_progress_reporter(
        Some((vocab_size-current_vocab_size) as usize), //length
        rank,
        "🤝 merging pairs",
        None,
        1000, // interval
        true  // verbose
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

    progress.finish_with_message("🤝 pairs merged");
    
    Ok(merges)
}