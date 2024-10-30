mod datastructures;

use datastructures::{PairCounter,Block};

use std::collections::HashMap;
use std::time::Instant;

use rayon::prelude::*;
use counter::Counter;
use mpi::initialize;
use mpi::topology::Communicator;
use pyo3::prelude::*;


#[pyfunction]
pub fn train(all_blocks: Vec<Vec<u8>>, vocab_size: u32) -> HashMap<(u32, u32), u32> {

    // Init comms
    let universe = initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    
    // Log times
    let start = Instant::now();

    if rank == 0 {
        println!("Running BPE training algorithm...");
    }

    
    // Extract unique blocks and their counts
    let block_counter: Counter<Vec<u8>> = all_blocks.into_iter().collect();
    let mut blocks: Vec<Block> = Vec::new();
    for (_idx, (block, count)) in block_counter.into_iter().enumerate() {
        blocks.push(Block::new(block,count as i32));
    }

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
    
    merges
}