mod datastructures;
mod comms;

use datastructures::{PairCounter,Block};
use comms::reduce_block_counts;
use crate::utils::progress::{Progress,ProgressIteratorExt};

use std::collections::HashMap;

use rayon::prelude::*;
use mpi::initialize;
use mpi::topology::Communicator;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyString};

fn init_datastructures(
    generator: &Bound<'_, PyIterator>,
) -> Option<(Vec<Block>,PairCounter)> {
    // Init comms
    let universe = initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();

    if rank == 0 {
        println!("Running BPE training algorithm...");
    }

    // Count the duplicate blocks in the generator
    let block_count_progress = Progress::new(
        None,      // length
        rank,
        "🧱 counting blocks",
        Some("🧱 blocks counted")
    );

    let mut local_block_counts: HashMap<String, i32> = HashMap::new();

    generator
        .into_iter()
        .attach_progress(block_count_progress)
        .for_each(|s| {
            local_block_counts.entry(
                s.unwrap()
                .downcast::<PyString>().unwrap()
                .to_str().unwrap()
                .to_string()
            ).and_modify(|c|*c+=1)
            .or_insert(1);
        });

    let block_counts = match reduce_block_counts(&world, local_block_counts) {
        Some(map) => map, // rank 0
        None => HashMap::new() // other ranks
    };

    // Convert block_counts to blocks
    let block_tokenize_progress = Progress::new(
        Some(block_counts.len()), // length
        rank,
        "🔢 tokenizing blocks",
        Some("🔢 blocks tokenized")
    );

    let blocks: Vec<Block> = block_counts
        .into_iter()
        .attach_progress(block_tokenize_progress)
        .map(|(s, count)| {
            Block::new(s, count as i32)
        })
        .collect();

    // Init pair counter
    let bp_counts = PairCounter::new(&blocks,&world); 

    match rank {
        0 => Some((blocks, bp_counts)),
        _ => None
    }
} 


#[pyfunction]
pub fn train(
    generator: &Bound<'_, PyIterator>, 
    vocab_size: u32, 
) -> PyResult<HashMap<(u32, u32), u32>> {

    if let Some((blocks, mut bp_counts)) = init_datastructures(generator) {

        // Begin training
        let mut current_vocab_size: u32 = 256;
        let mut merges: HashMap<(u32, u32), u32> = HashMap::new(); 
        
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

            merges.insert(pair.vals, current_vocab_size);
            
            let changes: HashMap<(u32, u32), (i32, Vec<usize>)> = pair.block_ids
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
                });

                // Commit changes and sync across ranks
                bp_counts.commit(changes);
                
                current_vocab_size += 1;
                
                progress.inc(1);
                //println!("New bytepair merge {:?} -> {:?} with count {:?}.", pair.vals, current_vocab_size, pair.count);
            }

            progress.finish();
            Ok(merges)
        } else {
            Ok(HashMap::new())
        }
    }