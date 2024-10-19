use mpi::traits::*;
use std::collections::HashSet;

// Import the necessary structs and functions from your library
use rustbpe::{PairHeap,Block,all_reduce_counts};


#[test]
fn test_add() {
    // Initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    // Define blocks for each process
    // For simplicity, each process has unique tokens
    let blocks = match rank {
        0 => vec![
            Block { tokens: vec![1, 2, 3, 4] },
            Block { tokens: vec![2, 3, 5] },
        ],
        1 => vec![
            Block { tokens: vec![1, 2, 3] },
            Block { tokens: vec![3, 4, 5] },
        ],
        2 => vec![
            Block { tokens: vec![1, 3, 5] },
            Block { tokens: vec![2, 4, 6] },
        ],
        3 => vec![
            Block { tokens: vec![1, 2, 6] },
            Block { tokens: vec![3, 5, 6] },
        ],
        _ => vec![], // Additional processes if any
    };

    // Initialize PairHeap
    let mut pair_heap = PairHeap::new(blocks, &world);

    // Perform some operations
    // For demonstration, each process adds and removes some pairs
    // The actual operations should be based on your test scenarios

    // Example: Each process adds a specific pair
    pair_heap.add((10, 20), 5, 1 as usize);

    // Commit the changes (this should perform the all_reduce operations)
    pair_heap.commit();

    // Synchronize all processes
    world.barrier();

    // Now, validate the results
    // Since all_reduce_counts and all_reduce_removes are placeholders,
    // you need to implement the actual reduction logic to perform correct aggregation.

    // For demonstration, we'll perform a simple check on the counts of added pairs
    // In a real scenario, you should compute the expected global counts and compare

    // Example: Check that (10, 20) has a total count of 5 * size
    let expected_count_10_20 = 5 * size;
    let key_10_20 = (10, 20);
    let exists_10_20 = pair_heap.d.contains_key(&key_10_20);

    if rank == 0 {
        // Rank 0 performs the assertions
        assert!(exists_10_20, "Pair (10, 20) should exist in the PairHeap.");
        let pair_idx = pair_heap.d.get(&key_10_20).unwrap();
        let pair = &pair_heap.pairs[*pair_idx];
        assert_eq!(
            pair.count, expected_count_10_20,
            "Pair (10, 20) should have a count of {}",
            expected_count_10_20
        );
    }
}

#[test]
fn test_most_common() {
    // Initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    // Define blocks for each process
    let blocks = match rank {
        0 => vec![
            Block { tokens: vec![1, 2, 3, 4] },
            Block { tokens: vec![2, 3, 5] },
        ],
        1 => vec![
            Block { tokens: vec![1, 2, 3] },
            Block { tokens: vec![3, 4, 5] },
        ],
        2 => vec![
            Block { tokens: vec![1, 3, 5] },
            Block { tokens: vec![2, 4, 6] },
        ],
        3 => vec![
            Block { tokens: vec![1, 2, 6] },
            Block { tokens: vec![3, 5, 6] },
        ],
        _ => vec![],
    };

    // Initialize PairHeap
    let mut pair_heap = PairHeap::new(blocks, &world);

    // Perform operations
    // Each process adds the same pair multiple times to ensure it's the most common
    pair_heap.add((100, 200), 10, 0 as usize); // This should be the most common
    pair_heap.add((200, 300), 5, 1 as usize);

    // Get the most common pair
    let most_common = pair_heap.most_common();

    // Synchronize all processes
    world.barrier();

    // Validate that the most common pair is (100, 200) with the expected count
    if rank == 0 {
        assert!(
            most_common.is_some(),
            "There should be a most common pair in the PairHeap."
        );
        let ((left, right), count) = most_common.unwrap();
        assert_eq!(
            (left, right),
            (100, 200),
            "The most common pair should be (100, 200)."
        );
        let expected_count = 10 * size;
        assert_eq!(
            count, expected_count,
            "The count for (100, 200) should be {}.",
            expected_count
        );
    }
}

#[test]
fn test_remove_all() {
    // Initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    // Define blocks for each process
    let blocks = match rank {
        0 => vec![
            Block { tokens: vec![1, 2, 3, 4] },
            Block { tokens: vec![2, 3, 5] },
        ],
        1 => vec![
            Block { tokens: vec![1, 2, 3] },
            Block { tokens: vec![3, 4, 5] },
        ],
        2 => vec![
            Block { tokens: vec![1, 3, 5] },
            Block { tokens: vec![2, 4, 6] },
        ],
        3 => vec![
            Block { tokens: vec![1, 2, 6] },
            Block { tokens: vec![3, 5, 6] },
        ],
        _ => vec![],
    };

    // Initialize PairHeap
    let mut pair_heap = PairHeap::new(blocks, &world);

    // Each process adds a specific pair
    pair_heap.add((50, 60), 4, 0 as usize);

    // Commit the additions
    pair_heap.commit();

    // Now, each process removes all instances of (50, 60)
    pair_heap.remove((50, 60), 4);

    // Commit the removals
    pair_heap.commit();

    // Synchronize all processes
    world.barrier();

    // Validate that (50, 60) has been removed from the PairHeap
    if rank == 0 {
        let key = (50, 60);
        assert!(
            !pair_heap.d.contains_key(&key),
            "Pair (50, 60) should have been removed from the PairHeap."
        );
    }
}
