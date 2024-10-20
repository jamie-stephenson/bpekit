use crate::pairheap::PairHeap;

use std::time::Instant;

use mpi::initialize;
use mpi::topology::Communicator;
use pyo3::prelude::*;

#[pyfunction]
pub fn bpe(mut blocks: Vec<Vec<u32>>, vocab_size: u32) -> (Vec<((u32, u32), u32)>,Vec<Vec<u32>>) {

    let universe = initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    let start = Instant::now();

    if rank == 0 {
        println!("Running BPE algorithm...");
    }
   
    let mut bp_counts = PairHeap::new(&blocks,&world);    

    if rank == 0 {
        let init_duration = start.elapsed();
        println!("Initialising datastuctures took {:.2} seconds", init_duration.as_secs());
    }

    let mut current_vocab_size: u32 = 256;
    let mut merges: Vec<((u32, u32), u32)> = Vec::new(); 

    while current_vocab_size < vocab_size {

        let ((left,right),count,block_ids) = match bp_counts.most_common() { 
            Some(tuple) => tuple,
            None => break // Exit the loop if no pairs are left
        };

        if rank == 0 {
            println!("New bytepair merge {:?} -> {:?} with count {:?}.", (left,right), current_vocab_size, count);
        }

        merges.push(((left,right), current_vocab_size));

        for block_idx in block_ids {

            let block = &mut blocks[block_idx];
            let mut token_idx = 0;

            while token_idx < block.len() {
                
                if block[token_idx] == left && token_idx + 1 < block.len() && block[token_idx+1] == right {

                    bp_counts.remove((left,right), 1);
       
                    // Handle the previous token if it exists
                    if token_idx > 0 {
                        let prev_token = block[token_idx-1];
                        bp_counts.remove((prev_token, left), 1);                                    
                        bp_counts.add((prev_token, current_vocab_size), 1, block_idx);
                    }
                    
                    block[token_idx] = current_vocab_size;
                    block.remove(token_idx+1);
                    
                    // Handle the next token if it exists
                    if token_idx + 1 < block.len() {
                        let next_token = block[token_idx+1];
                        bp_counts.remove((right, next_token), 1);                                             
                        bp_counts.add((current_vocab_size, next_token), 1, block_idx);
                    }
                }
                token_idx += 1;
            }
        }
        current_vocab_size += 1;
    }

    if rank == 0 {
        let duration = start.elapsed();
        println!("BPE algorithm complete in {:.2} seconds", duration.as_secs());
    }
    
    (merges, blocks)
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

    #[test]
    fn test_bpe_triple_token() {
        // Create a block with three identical u32 values in a row
        let blocks = vec![vec![108, 108, 108]];

        // Set a vocab size that allows for at least one merge
        let vocab_size = 258;

        // Call the bpe function and get the result
        let (merges, _blocks) = bpe(blocks.clone(), vocab_size);

        assert_eq!(merges[0].0, (108, 108), "Expected to merge the pair (108, 108).");
        assert_eq!(merges[0].1, 256, "Expected the first new vocab entry to be 256.");
        assert_eq!(merges[1].0, (256, 108), "Expected to merge the pair (256, 108).");
        assert_eq!(merges[1].1, 257, "Expected the second new vocab entry to be 257.");

    }

}

