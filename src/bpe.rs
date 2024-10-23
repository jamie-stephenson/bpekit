use crate::paircounter::PairCounter;

use std::collections::{HashMap,HashSet};
use std::time::Instant;

use rayon::prelude::*;
use counter::Counter;
use mpi::initialize;
use mpi::topology::Communicator;
use pyo3::prelude::*;

fn perform_merge_in_block(
    left: u32,
    right:u32,
    new: u32, 
    block_idx: usize,
    block: &mut Vec<u32>
) -> HashMap<(u32,u32),(i32,Vec<usize>)> {

    let mut changes: HashMap<(u32,u32),(i32,Vec<usize>)> = HashMap::new();
    let mut token_idx = 0;

    while token_idx < block.len() {
        
        if block[token_idx] == left && token_idx + 1 < block.len() && block[token_idx+1] == right {

            changes
                .entry((left, right))
                .and_modify(|(change,idx)| *change -= 1)
                .or_insert((-1,vec![]));

            // Handle the previous token if it exists
            if token_idx > 0 {
                let prev_token = block[token_idx-1];
                changes
                    .entry((prev_token, left))
                    .and_modify(|(change,_idx)| *change -= 1)
                    .or_insert((-1,vec![]));
                changes
                    .entry((prev_token, new))
                    .and_modify(|(change,_idx)| *change += 1)
                    .or_insert((1,vec![block_idx]));
            }
            
            block[token_idx] = new;
            block.remove(token_idx+1);
            
            // Handle the next token if it exists
            if token_idx + 1 < block.len() {
                let next_token = block[token_idx+1]; 
                changes
                    .entry((right, next_token))
                    .and_modify(|(change,_idx)| *change -= 1)
                    .or_insert((-1,vec![]));
                changes
                    .entry((new, next_token))
                    .and_modify(|(change,_idx)| *change += 1)
                    .or_insert((1,vec![block_idx]));
            }
        }
        token_idx += 1;
    }
    changes
}


#[pyfunction]
pub fn bpe(all_blocks: Vec<Vec<u8>>, vocab_size: u32) -> Vec<((u32, u32), u32)> {

    // Init comms
    let universe = initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    
    // Log times
    let start = Instant::now();

    if rank == 0 {
        println!("Running BPE algorithm...");
    }

    
    // Extract unique blocks and index their counts
    let block_counter = all_blocks.into_iter().collect::<Counter<_>>();
    let mut blocks: Vec<Vec<u32>> = Vec::new();
    let mut block_idx_counts: HashMap<usize, i32> = HashMap::new();
    for (index, (block, count)) in block_counter.into_iter().enumerate() {
        blocks.push(
            block
                .into_iter()
                .map(|byte| byte as u32)
                .collect()
        );
        block_idx_counts.insert(index, count as i32);
    }

    // Init pair counter
    let mut bp_counts = PairCounter::new(&blocks,&block_idx_counts,&world);    

    // Log times
    if rank == 0 {
        let init_duration = start.elapsed();
        println!("Initialising datastuctures took {:.2} seconds", init_duration.as_secs());
    }

    // Begin training
    let mut current_vocab_size: u32 = 256;
    let mut merges: Vec<((u32, u32), u32)> = Vec::new(); 

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

        merges.push((pair.vals, current_vocab_size));

        let changes: Vec<((u32, u32), (i32, Vec<usize>))> = pair.block_ids
            .par_iter()
            .map(|&block_idx| {
                let block_ptr = &blocks[block_idx] as *const _ as *mut Vec<u32>;
                unsafe {
                    perform_merge_in_block(
                        left, 
                        right, 
                        current_vocab_size,
                        block_idx, 
                        &mut *block_ptr
                    )
                }
            })
            .reduce(
                || HashMap::new(),
                |mut changes1, changes2| {
                    for (pair, (change, block_ids)) in changes2 {
                        changes1
                            .entry(pair)
                            .and_modify(|(change,ids)| {
                                *change += 1;
                                ids.append(&mut block_ids);
                            })                             
                            .or_insert((change,block_ids));
                    }
                    changes1
                },
            ).into_iter().collect();

        bp_counts.

        current_vocab_size += 1;
    }

    if rank == 0 {
        let duration = start.elapsed();
        println!("BPE algorithm complete in {:.2} seconds", duration.as_secs());
    }
    
    merges
}

#[cfg(test)]
mod tests {

    use super::*;

    // #[test]
    // fn test_bpe_basic_merge() {
    //     let blocks = vec![
    //         vec![1, 2, 3, 1, 2],
    //         vec![3, 1, 2, 4, 1, 2],
    //     ];

    //     let vocab_size = 258; // Set a small vocab size for testing

    //     // Run the BPE algorithm and capture the merges
    //     let (merges, _blocks) = bpe(blocks,vocab_size);

    //     // Debugging output
    //     println!("Merges: {:?}", merges);

    //     // Check that the correct merge was performed
    //     assert_eq!(merges[0].0, (1, 2)); 
    //     assert_eq!(merges[0].1, 256);
    //     assert_eq!(merges[1].0, (3, 256)); 
    //     assert_eq!(merges[1].1, 257);    

    //     // Ensure that the merge list has the expected number of entries
    //     assert_eq!(merges.len(), 2); // As vocab_size is 258, we should see two merges
    // }

    // #[test]
    // fn test_bpe_empty_input() {
    //     let blocks: Vec<Vec<u32>> = vec![]; // Empty input
    //     let vocab_size = 300;

    //     // Run the BPE algorithm and capture the merges
    //     let (merges, _blocks) = bpe(blocks,vocab_size);

    //     // In case of empty input, we expect no merges to happen
    //     assert!(merges.is_empty());
    // }

    // #[test]
    // fn test_bpe_stop_at_vocab_size() {
    //     let blocks = vec![
    //         vec![5, 6, 7, 8],
    //         vec![5, 7, 8, 9, 6, 7],
    //     ];

    //     let vocab_size = 260;

    //     // Run the BPE algorithm and capture the merges
    //     let (merges, _blocks) = bpe(blocks,vocab_size);

    //     // Debugging output
    //     println!("Merges: {:?}", merges);

    //     // Ensure that the number of merges performed is consistent with the vocab_size limit
    //     assert_eq!(merges.len(), 4); // Expect 4 merges, as initial vocab size is 256
    // }

    // #[test]
    // fn test_bpe_single_block_merge() {
    //     let blocks = vec![
    //         vec![10, 11, 12, 10, 11, 12, 10, 11],
    //     ];

    //     let vocab_size = 260;

    //     // Run the BPE algorithm and capture the merges
    //     let (merges, _blocks) = bpe(blocks,vocab_size);

    //     // Debugging output
    //     println!("Merges: {:?}", merges);

    //     // Check that the most common pair (10, 11) was merged
    //     assert_eq!(merges[0].0, (10, 11)); // First merge should be (10, 11)
    //     assert_eq!(merges[0].1, 256);      // New token assigned to the merge should be 256

    //     // Ensure that more merges have been performed until reaching the vocab size limit
    //     assert!(merges.len() > 0);
    // }

    // #[test]
    // fn test_bpe_triple_token() {
    //     // Create a block with three identical u32 values in a row
    //     let blocks = vec![vec![108, 108, 108]];

    //     // Set a vocab size that allows for at least one merge
    //     let vocab_size = 258;

    //     // Call the bpe function and get the result
    //     let merges = bpe(blocks.clone(), vocab_size);

    //     assert_eq!(merges[0].0, (108, 108), "Expected to merge the pair (108, 108).");
    //     assert_eq!(merges[0].1, 256, "Expected the first new vocab entry to be 256.");
    //     assert_eq!(merges[1].0, (256, 108), "Expected to merge the pair (256, 108).");
    //     assert_eq!(merges[1].1, 257, "Expected the second new vocab entry to be 257.");

    // }

}

