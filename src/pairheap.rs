use crate::all_reduce_counts::all_reduce_counts;
use std::collections::{HashMap,HashSet};
use std::fmt;
use std::cmp::min;

use mpi::topology::SimpleCommunicator;

#[derive(Debug, Clone)]
struct Pair {
    count: u64,               // Global count
    vals: (u32, u32),
    heap_pos: usize,
    block_ids: HashSet<usize> // Indices of local blocks containing this pair
}

// struct to maintain counts across all processes
pub struct PairHeap<'a> {
    pairs: Vec<Pair>,                                       // Each pair contains ids for local blocks
    heap: Vec<usize>,                                       // Heap of indices into pairs, maintained based on global counts
    d: HashMap<(u32, u32), usize>,                          // Map from pair to pair index
    to_add: HashMap<(u32, u32), u64>,                       // Pending additions
    to_add_block_ids: HashMap<(u32, u32), HashSet<usize>>,  // Local block ids of the pending additions
    to_remove: HashMap<(u32, u32), u64>,                    // Pending removals
    world: &'a SimpleCommunicator                           // For collective comms
}

impl<'a> PairHeap<'a> {

    pub fn new(data: &[Vec<u32>], world: &'a SimpleCommunicator) -> Self {
        let mut ph = PairHeap {
            pairs: Vec::new(),
            heap: Vec::new(),
            d: HashMap::new(),
            to_add: HashMap::new(),
            to_add_block_ids: HashMap::new(),
            to_remove: HashMap::new(),
            world: world
        };

        // Process each block to extract pairs and add them to the heap
        for block in data {
            // Extract adjacent pairs
            if block.len() >= 2 {
                for pair in block.windows(2) {
                    let a = pair[0];
                    let b = pair[1];
                    ph.add((a, b), 1);
                }
            }
        }

        // Commit pending additions
        ph.commit();

        ph
    }

    pub fn add(&mut self, item: (u32, u32), count: u64) {
        *self.to_add.entry(item).or_insert(0) += count;
    }


    // Remove an item with a count
    pub fn remove(&mut self, item: (u32, u32), count: u64) {
        *self.to_remove.entry(item).or_insert(0) += count;
    }

    // Commit pending additions and removals
    fn commit(&mut self) {

        // NOTE: Hashmaps aren't ordered so draining them to an iterator here causes non-deterministic behaviour.
        // `all_reduce_counts` has extra logic to ensure that all processes adjust the heap in the same way
        
        let to_add_local: Vec<((u32, u32), u64)> = self.to_add.drain().collect();
        let to_remove_local: Vec<((u32, u32), u64)> = self.to_remove.drain().collect();

        let to_add_global = all_reduce_counts(&self.world, to_add_local);
        let to_remove_global = all_reduce_counts(&self.world, to_remove_local);

        // Process additions
        for (item, count) in to_add_global {
            let block_ids = self.to_add_block_ids
                                .get(&item)
                                .cloned()
                                .unwrap_or_else(HashSet::new);
            self.add_pair(item, count, block_ids);
        }

        // Process removals
        for (item, count) in to_remove_global {
            self.remove_pair(item, count);
        }
    }

    // Internal method to add items
    fn add_pair(&mut self, item: (u32, u32), count: u64, block_ids: HashSet<usize>) {
        if count == 0 {
            return;
        }
        if let Some(&pair_idx) = self.d.get(&item) {
            // To avoid multiple mutable borrows, extract heap_pos first
            let heap_pos = {
                let pair = &mut self.pairs[pair_idx];
                pair.count += count;
                pair.block_ids.extend(&block_ids);
                pair.heap_pos
            };
            self.item_increased(heap_pos);
        } else {
            let pair_idx = self.pairs.len();
            let heap_pos = self.heap.len();
            let pair = Pair{
                count: count,
                vals: item,
                heap_pos: heap_pos,
                block_ids: block_ids
            };
            self.pairs.push(pair);
            self.d.insert(item, pair_idx);
            self.heap.push(pair_idx);
            self.item_increased(heap_pos);
        }
    }

    // Internal method to remove items
    fn remove_pair(&mut self, item: (u32, u32), count: u64) {
        if count == 0 {
            return;
        }
        if let Some(&node_idx) = self.d.get(&item) {
            // Extract heap_pos and update count safely
            let heap_pos;
            {
                let node = &mut self.pairs[node_idx];

                
                node.count -= count;
                heap_pos = node.heap_pos;
            }
            self.item_decreased(heap_pos);
        }
    }

    // Get the most common item, its count and its block ids 
    pub fn most_common(&mut self) -> Option<((u32,u32),u64,HashSet<usize>)> {
        self.commit();
        if !self.heap.is_empty() {
            let pair_idx = self.heap[0];
            let pair = self.pairs[pair_idx].clone(); 
            Some((pair.vals,pair.count,pair.block_ids))
        } else {
            None
        }
    }

    // Maintain the heap when an item's count increases
    fn item_increased(&mut self, mut pos: usize) {
        let node_idx = self.heap[pos];
        while pos > 0 {
            let parent_pos = (pos - 1) / 2;
            let parent_idx = self.heap[parent_pos];
            // Compare counts directly
            if self.pairs[parent_idx].count < self.pairs[node_idx].count {
                // Swap positions
                self.heap[pos] = parent_idx;
                self.pairs[parent_idx].heap_pos = pos;
                pos = parent_pos;
            } else {
                break;
            }
        }
        self.heap[pos] = node_idx;
        self.pairs[node_idx].heap_pos = pos;
    }

    // Maintain the heap when an item's count decreases
    fn item_decreased(&mut self, mut pos: usize) {
        let len = self.heap.len();
        let node_idx = self.heap[pos];

        // `child` starts and ends each loop pointing to the left child
        // but during the loop it points to the largest child
        let mut child: usize = 2*pos + 1; 
        let mut child_idx: usize;

        let mut right_child: usize;
        let mut right_child_idx: usize;

        while child < len {

            child_idx = self.heap[child];
            right_child = child + 1;
            
            if right_child < len {

                right_child_idx = self.heap[right_child];

                if self.pairs[child_idx].count < self.pairs[right_child_idx].count {

                    child = right_child; // make sure `child` points to largest child
                    child_idx = right_child_idx;

                }
            }
            
            if self.pairs[node_idx].count < self.pairs[child_idx].count {
                // Swap positions
                self.heap[pos] = child_idx;
                self.pairs[child_idx].heap_pos = pos;
                pos = child;
                child = 2*pos + 1; // make sure `child` points to left child
            } else {
                break;
            }
        }
    
        self.heap[pos] = node_idx;
        self.pairs[node_idx].heap_pos = pos;
    }
    
}

impl<'a> fmt::Debug for PairHeap<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.heap.is_empty() {
            writeln!(f, "PairHeap is empty.")
        } else {
            self.print_heap(f)
        }
    }
}

impl<'a> PairHeap<'a> {
    fn print_heap(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let heap_size = self.heap.len();
        let height = (heap_size as f64).log2().ceil() as usize;
        let max_width = 2usize.pow(height as u32) * 4; // Adjust width per node if needed

        for level in 0..height {
            let level_pairs = 2usize.pow(level as u32);
            let indent_space = max_width / (level_pairs * 2);
            let between_space = max_width / level_pairs - indent_space * 2;
            let start = level_pairs - 1;
            let end = min(start + level_pairs, heap_size);

            // Print pairs
            let mut line = String::new();
            for i in start..end {
                if i == start {
                    line.push_str(&" ".repeat(indent_space));
                } else {
                    line.push_str(&" ".repeat(between_space));
                }
                let pair_idx = self.heap[i];
                let pair = &self.pairs[pair_idx];
                line.push_str(&format!("({},{})", pair.vals.0, pair.vals.1));
            }
            writeln!(f, "{}", line)?;

            // Print branches
            if level < height - 1 {
                let mut branch_line = String::new();
                for i in start..end {
                    if i == start {
                        branch_line.push_str(&" ".repeat(indent_space));
                    } else {
                        branch_line.push_str(&" ".repeat(between_space));
                    }
                    let left_child = 2 * i + 1;
                    let right_child = 2 * i + 2;

                    if left_child < heap_size {
                        branch_line.push('/');
                    } else {
                        branch_line.push(' ');
                    }

                    branch_line.push_str(&" ".repeat(3)); // Adjust if pairs have wider representation

                    if right_child < heap_size {
                        branch_line.push('\\');
                    } else {
                        branch_line.push(' ');
                    }
                }
                writeln!(f, "{}", branch_line)?;
            }
        }

        Ok(())
    }
}