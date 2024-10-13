use crate::indexed_blocks::IndexedBlocks;
use crate::multiset::DistributedMultiset;

use std::time::Instant;

use mpi::initialize;
use mpi::topology::Communicator;
use pyo3::prelude::*;

#[pyfunction]
pub fn bpe(blocks: Vec<Vec<u32>>, vocab_size: u32) -> (Vec<((u32, u32), u32)>,Vec<Vec<u32>>) {

    let universe = initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    let start = Instant::now();

    if rank == 0 {
        println!("Running BPE algorithm...");
    }
   
    let mut indexed_blocks = IndexedBlocks::new(blocks.clone()); 
    let mut bp_counts = DistributedMultiset::new(blocks.clone(),&world);    

    if rank == 0 {
        let init_duration = start.elapsed();
        println!("Initialising datastuctures took {:.2} seconds", init_duration.as_secs());
    }

    let mut current_vocab_size: u32 = 256;
    let mut merges: Vec<((u32, u32), u32)> = Vec::new(); 

    while current_vocab_size < vocab_size {

        let (pair_to_merge,count) = match bp_counts.most_common() { 
            Some(pair) => pair,
            None => break // Exit the loop if no pairs are left
        };

        if rank == 0 {
            println!("New bytepair merge {:?} -> {:?} with count {:?}.", pair_to_merge, current_vocab_size, count);
        }

        merges.push((pair_to_merge, current_vocab_size)); 

        // Get the list of nodes corresponding to 'pair_to_merge'
        if let Some(nodes) = indexed_blocks.index.get(&pair_to_merge) {

            let nodes_vec: Vec<_> = nodes.iter().filter_map(|weak_node| weak_node.upgrade()).collect();

            for node in nodes_vec {
                
                // First, borrow node immutably for the initial checks
                let proceed = {
                    let node_ref = node.borrow();
                    
                    // Check if node.val matches
                    if node_ref.val != pair_to_merge.0 {
                        false
                    } else if let Some(next_rc) = &node_ref.next {
                        let next_node_ref = next_rc.borrow();
                        if next_node_ref.val != pair_to_merge.1 {
                            false
                        } else {
                            true
                        }
                    } else {
                        false
                    }
                };

                if !proceed {
                    continue;
                }

                // Remove the pair from bp_counts
                bp_counts.remove(pair_to_merge, 1);

                // Handle the next-next node if it exists
                if let Some(next_rc) = node.borrow().next.clone() {
                    let next_node_ref = next_rc.borrow();
                    if let Some(next_next_rc) = &next_node_ref.next {
                        let next_next_node_ref = next_next_rc.borrow();
                        let next_val = next_node_ref.val;
                        let next_next_val = next_next_node_ref.val;
                        bp_counts.remove((next_val, next_next_val), 1);                                             
                        bp_counts.add((current_vocab_size, next_next_val), 1);                                     
                    }
                }

                // Handle the previous node if it exists
                if let Some(prev_weak) = node.borrow().prev.clone() {
                    if let Some(prev_rc) = prev_weak.upgrade() {
                        let prev_node_ref = prev_rc.borrow();
                        bp_counts.remove((prev_node_ref.val, node.borrow().val), 1);                                    
                        bp_counts.add((prev_node_ref.val, current_vocab_size), 1);                      
                    }
                }

                // To update node.val we need to mutably borrow node
                // To delete node.next we need to mutably borrow node.next.prev == node
                // Therefore we need two mutable references to the same node: Not allowed!
                // This means we have to carefully do one and then the other in different scopes.
                let next_rc_opt = {
                    let mut node_ref = node.borrow_mut(); 
                    let next_rc = node_ref.next.take();
                    node_ref.val = current_vocab_size;
                    next_rc 
                }; 

                if let Some(next_rc) = next_rc_opt {
                    next_rc.borrow_mut().delete(); 
                }

                indexed_blocks.update_index(&node); 
            }
        }
        current_vocab_size += 1;
    }

    if rank == 0 {
        let duration = start.elapsed();
        println!("BPE algorithm complete in {:.2} seconds", duration.as_secs());
    }
    
    (merges, indexed_blocks.drain_blocks())
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_bpe_basic_merge() {
        let blocks = vec![
            vec![1, 2, 3, 1, 2],
            vec![3, 1, 2, 4, 1, 2],
        ];

        let vocab_size = 258; // Set a small vocab size for testing

        // Run the BPE algorithm and capture the merges
        let (merges, _blocks) = bpe(blocks,vocab_size);

        // Debugging output
        println!("Merges: {:?}", merges);

        // Check that the correct merge was performed
        assert_eq!(merges[0].0, (1, 2)); 
        assert_eq!(merges[0].1, 256);
        assert_eq!(merges[1].0, (3, 256)); 
        assert_eq!(merges[1].1, 257);    

        // Ensure that the merge list has the expected number of entries
        assert_eq!(merges.len(), 2); // As vocab_size is 258, we should see two merges
    }

    #[test]
    fn test_bpe_empty_input() {
        let blocks: Vec<Vec<u32>> = vec![]; // Empty input
        let vocab_size = 300;

        // Run the BPE algorithm and capture the merges
        let (merges, _blocks) = bpe(blocks,vocab_size);

        // In case of empty input, we expect no merges to happen
        assert!(merges.is_empty());
    }

    #[test]
    fn test_bpe_stop_at_vocab_size() {
        let blocks = vec![
            vec![5, 6, 7, 8],
            vec![5, 7, 8, 9, 6, 7],
        ];

        let vocab_size = 260;

        // Run the BPE algorithm and capture the merges
        let (merges, _blocks) = bpe(blocks,vocab_size);

        // Debugging output
        println!("Merges: {:?}", merges);

        // Ensure that the number of merges performed is consistent with the vocab_size limit
        assert_eq!(merges.len(), 4); // Expect 4 merges, as initial vocab size is 256
    }

    #[test]
    fn test_bpe_single_block_merge() {
        let blocks = vec![
            vec![10, 11, 12, 10, 11, 12, 10, 11],
        ];

        let vocab_size = 260;

        // Run the BPE algorithm and capture the merges
        let (merges, _blocks) = bpe(blocks,vocab_size);

        // Debugging output
        println!("Merges: {:?}", merges);

        // Check that the most common pair (10, 11) was merged
        assert_eq!(merges[0].0, (10, 11)); // First merge should be (10, 11)
        assert_eq!(merges[0].1, 256);      // New token assigned to the merge should be 256

        // Ensure that more merges have been performed until reaching the vocab size limit
        assert!(merges.len() > 0);
    }

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

