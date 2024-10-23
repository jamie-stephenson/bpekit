use crate::all_reduce_counts::all_reduce_counts;
use std::collections::{HashMap,HashSet,BinaryHeap};
use std::cmp::Ordering;

use mpi::topology::SimpleCommunicator;

#[derive(Debug, Eq)]
pub(crate) struct Pair {
    pub count: i32,               // Global count
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
    world: &'a SimpleCommunicator                           // For collective comms
}

impl<'a> PairCounter<'a> {

    pub fn new(blocks: &[Vec<u32>], block_idx_counts: &HashMap<usize,i32>, world: &'a SimpleCommunicator) -> Self {
        let mut pc = PairCounter {
            heap: BinaryHeap::new(),
            counts: HashMap::new(),
            world: world
        };

        // Process each block to extract pairs and add them to the heap
        for (block_idx,block) in blocks.into_iter().enumerate() {

            let block_count = block_idx_counts[&block_idx]; 
            
            // Extract adjacent pairs
            if block.len() >= 2 {
                for pair in block.windows(2) {
                    let a = pair[0] as u32;
                    let b = pair[1] as u32;
                    pc.change((a, b), block_count);
                    pc.add_block_idx((a,b), block_idx);
                }
            }
        }
        // Commit pending changes
        pc.commit();
        pc
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
    fn commit(&mut self, changes: Vec<((u32,u32),(i32,Vec<usize>))>) {
        
        let to_change_global = all_reduce_counts(&self.world, changes); 

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