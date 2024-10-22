use crate::all_reduce_counts::all_reduce_counts;
use std::collections::{HashMap,HashSet,BinaryHeap};
use std::cmp::Ordering;

use mpi::topology::SimpleCommunicator;

#[derive(Debug, Eq)]
pub(crate) struct Pair {
    pub count: i32,             // Global count
    pub vals: (u32, u32),
    pub block_ids: HashSet<usize> // Indices of local blocks containing this pair
}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.vals == other.vals && self.vals == other.vals
    }
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count.cmp(&other.count)
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// struct to maintain counts across all processes
pub(crate) struct PairCounter<'a> {
    heap: BinaryHeap<Pair>,                                 // Heap of pairs, maintained based on global counts
    counts: HashMap<(u32, u32), i32>,                       // Map from pair to global count, used to validate the top of the heap
    to_change: HashMap<(u32, u32), i32>,                    // Pending changes
    to_add_block_ids: HashMap<(u32, u32), HashSet<usize>>,  // Local block ids of the pending additions
    world: &'a SimpleCommunicator                           // For collective comms
}

impl<'a> PairCounter<'a> {

    pub fn new(data: &[Vec<u32>], world: &'a SimpleCommunicator) -> Self {
        let mut ph = PairCounter {
            heap: BinaryHeap::new(),
            counts: HashMap::new(),
            to_change: HashMap::new(),
            to_add_block_ids: HashMap::new(),
            world: world
        };

        // Process each block to extract pairs and add them to the heap
        for (block_idx,block) in data.into_iter().enumerate() {
            // Extract adjacent pairs
            if block.len() >= 2 {
                for pair in block.windows(2) {
                    let a = pair[0];
                    let b = pair[1];
                    ph.change((a, b), 1);
                    ph.add_block_idx((a,b), block_idx);
                }
            }
        }
        // Commit pending changes
        ph.commit();
        ph
    }

    pub fn add_block_idx(&mut self, pair: (u32, u32), block_idx: usize) {
        self.to_add_block_ids
            .entry(pair)
            .or_insert(HashSet::new())
            .insert(block_idx);
    }

    pub fn change(&mut self, pair: (u32, u32), count: i32) {
        self.to_change
            .entry(pair)
            .and_modify(|c| *c += count)
            .or_insert(count);
    }

    pub fn pop(&mut self) -> Option<Pair> {
        self.commit();
        self.heap.pop()
    }

    // check if pair is stale
    pub fn is_stale(&mut self, pair: &Pair) -> bool {
        pair.count != self.counts[&pair.vals]
    }

    pub fn update_count_and_push(&mut self, mut pair: Pair) {
        pair.count = self.counts[&pair.vals];
        self.heap.push(pair)
    }    

    // Commit pending changes
    fn commit(&mut self) {

        // NOTE: Hashmaps aren't ordered so draining them to an iterator here causes non-deterministic behaviour.
        // `all_reduce_counts` has extra logic to ensure that all processes adjust the heap in the same way.
        
        let to_change_local: Vec<((u32, u32), i32)> = self.to_change.drain().collect();
        println!("Local changes {:?}",to_change_local);
        let to_change_global = all_reduce_counts(&self.world, to_change_local);


        // Process changes
        for (pair, change) in to_change_global {
            
            // You only ever have a positive change when you are introducing a new pair.
            // Add new pairs to the heap. 
            if change > 0 { 
                self.counts.insert(pair, change);

                let block_ids = self.to_add_block_ids
                .get(&pair)
                .cloned()
                .unwrap_or_else(HashSet::new);

                self.heap.push(Pair{count: change, vals: pair, block_ids});
            
            // Old pairs aren't removed from heap, 
            // instead we just reduce their count and lazily verify every iteration
            } else {
                self.counts.entry(pair).and_modify(|count| *count += change);
            }
        }
    } 
}


#[cfg(test)]
mod tests {

    use super::*;
    use mpi::topology::Communicator;

    /// Function to create your test data
    fn create_data() -> Vec<Vec<u32>> {
        vec![
            vec![1, 2, 3],
            vec![2, 3, 4],
            vec![3, 4, 5, 3, 4],
        ]
    }

    #[test]
    fn test_all() {
        // Initialize MPI once
        let universe = mpi::initialize().expect("Failed to initialize MPI");
        let world = universe.world();
        let size = world.size() as i32;

        // Test 1: PairCounter Initialization
        {
            // Each process initializes its data block
            let data = create_data();
            let mut ph = PairCounter::new(&data, &world);

            // On all ranks, check the most common pair after initialization
            if let Some(pair) = ph.pop() {
                assert_eq!(pair.vals, (3, 4), "Most common pair mismatch during initialization");
                assert_eq!(pair.count, 3 * size, "Count of (3,4) mismatch during initialization");
            } else {
                panic!("most_common() returned None unexpectedly during initialization");
            }
        }

        // Test 2: PairCounter Addition
        {
            let data = create_data();
            let mut ph = PairCounter::new(&data, &world);

            ph.change((4, 5), 1);

            if let Some(pair) = ph.pop() {
                assert_eq!(pair.vals, (3, 4), "Most common pair mismatch after first addition");
                assert_eq!(pair.count, 3 * size, "Count of (3,4) mismatch after first addition");
            } else {
                panic!("most_common() returned None unexpectedly after first addition");
            }

            ph.change((1, 2), 3);

            if let Some(pair) = ph.pop() {
                assert_eq!(pair.vals, (1, 2), "Most common pair mismatch after second addition");
                assert_eq!(pair.count, 4 * size, "Count of (1,2) mismatch after second addition");
            } else {
                panic!("most_common() returned None unexpectedly after second addition");
            }
        }

        // Test 3: PairCounter Removal
        {
            let data = create_data();
            let mut ph = PairCounter::new(&data, &world);

            ph.change((3, 4), -1); // Remove 1 occurrence of (3, 4)

            // After removing, (3,4) should still be the most common
            if let Some(pair) = ph.pop() {
                assert_eq!(pair.vals, (3, 4), "Most common pair mismatch after removal");
                assert_eq!(pair.count, 2*size, "Count of (3,4) mismatch after removal");
            } else {
                panic!("most_common() returned None unexpectedly after removal");
            }
        }

        // Test 4: Heap Structure
        {
            let data = create_data();
            let mut ph = PairCounter::new(&data, &world);

            ph.change((4, 5), 2);
            ph.pop();

            let rank = world.rank();

            // Print the heap structure (this is for visual inspection)
            if rank == 0 {
                println!("{:?}", ph.heap);
            }

            // We expect no panic, just visual confirmation of correct heap formatting
        }
    }
}