// Structures for maintaining pair counts during training

use crate::utils::progress::{Progress, ProgressIteratorExt};
use super::comms::reduce_block_counts;

use std::collections::{HashMap,HashSet,BinaryHeap};
use std::cmp::Ordering;

use rayon::prelude::*;
use mpi::topology::{SimpleCommunicator,Communicator};

#[derive(Debug, Eq)]
pub(crate) struct Pair {
    pub count: i32,           // Global count
    pub vals: (u32, u32),
    pub block_ids: Vec<usize> // Indices of local blocks containing this pair at time of pair creation
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
pub(crate) struct PairCounter {
    
    // Heap of pairs, maintained based on global counts
    heap: BinaryHeap<Pair>,                                 

    // Map from pair to global count, used to validate the top of the heap
    counts: HashMap<(u32, u32), i32>,                       
}

impl PairCounter {

    pub fn new(blocks: &[Block], world: &SimpleCommunicator) -> Self {
        let mut pc = PairCounter {
            heap: BinaryHeap::new(),
            counts: HashMap::new()
        };

        
        let progress = Progress::new(
            Some(blocks.len()),
            world.rank(),
            "🧮 counting pairs", 
            Some("🧮 pairs counted"),
        );

        let counts_map = blocks.into_iter()
            .enumerate()
            .attach_progress(progress)
            .par_bridge()
            .fold_with(HashMap::new(), |mut counts_map, (block_idx, block)| {
                if block.tokens.len() >= 2 {
                    for pair in block.tokens.windows(2) {
                        let a = pair[0] as u32;
                        let b = pair[1] as u32;
                        counts_map.entry((a, b))
                            .and_modify(|(count, set): &mut (i32, HashSet<usize>)| {
                                *count += block.count;
                                set.insert(block_idx);
                            })
                            .or_insert_with(|| {
                                let mut set = HashSet::new();
                                set.insert(block_idx);
                                (block.count, set)
                            });
                    }
                }
                counts_map
            })
            .reduce(|| HashMap::new(), |mut map1, map2| {
                // Merge map2 into map1
                for (key, (count2, set2)) in map2 {
                    map1.entry(key)
                        .and_modify(|(count1, set1)| {
                            *count1 += count2;
                            set1.extend(&set2);
                        })
                        .or_insert((count2, set2));
                }
                map1
            });
        
        // Convert to HashSet to Vec
        let counts: HashMap<(u32, u32), (i32, Vec<usize>)> = counts_map
            .into_iter()
            .map(|(pair, (count, block_ids))| {
                (pair,(count, block_ids.into_iter().collect()))
            }).collect();
        
        pc.commit(counts);

        pc
    }

    pub fn pop(&mut self) -> Option<Pair> {
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
    pub fn commit(&mut self, changes: HashMap<(u32,u32),(i32,Vec<usize>)>) {

        // Process changes
        for (pair, (change, block_ids)) in changes {
            
            // You only ever have a positive change when you are introducing a new pair.
            // Add new pairs to the heap. 
            if change > 0 { 
                self.counts.insert(pair, change);
                self.heap.push(Pair{count: change, vals: pair, block_ids});
            
            // Old pairs aren't removed from heap, 
            // instead we just reduce their count and lazily verify every iteration
            } else {
                self.counts.entry(pair).and_modify(|count| *count += change);
            }
        }
    } 
}

pub(crate) struct Block {
    pub tokens: Vec<u32>,
    pub count: i32
}

impl Block {
    pub fn new(s: String, count: i32) -> Block {
        Block{ 
            tokens: s.as_bytes()
                .into_iter()
                .map(|byte| *byte as u32)
                .collect(),
            count: count
        }
    }

    pub fn merge(
        &mut self, 
        left: u32, 
        right: u32, 
        new: u32,
        block_idx: usize
    ) -> HashMap<(u32,u32),(i32,Vec<usize>)> {
        let mut changes: HashMap<(u32,u32),(i32,Vec<usize>)> = HashMap::new();
        let mut token_idx = 0;

        while token_idx < self.tokens.len() {
            
            if self.tokens[token_idx] == left && token_idx + 1 < self.tokens.len() && self.tokens[token_idx+1] == right {

                changes
                    .entry((left, right))
                    .and_modify(|(change,_idx)| *change -= self.count)
                    .or_insert((-self.count,vec![]));

                // Handle the previous token if it exists
                if token_idx > 0 {
                    let prev_token = self.tokens[token_idx-1];
                    changes
                        .entry((prev_token, left))
                        .and_modify(|(change,_idx)| *change -= self.count)
                        .or_insert((-self.count,vec![]));
                    changes
                        .entry((prev_token, new))
                        .and_modify(|(change,_idx)| *change += self.count)
                        .or_insert((self.count,vec![block_idx]));
                }
                
                self.tokens[token_idx] = new;
                self.tokens.remove(token_idx+1);
                
                // Handle the next token if it exists
                if token_idx + 1 < self.tokens.len() {
                    let next_token = self.tokens[token_idx+1]; 
                    changes
                        .entry((right, next_token))
                        .and_modify(|(change,_idx)| *change -= self.count)
                        .or_insert((-self.count,vec![]));
                    changes
                        .entry((new, next_token))
                        .and_modify(|(change,_idx)| *change += self.count)
                        .or_insert((self.count,vec![block_idx]));
                }
            }
            token_idx += 1;
        }
        changes
    }
}

/// Initialize datastructures
pub fn init(
    local_blocks: impl Iterator<Item = String>
) -> (Vec<Block>,PairCounter) {

    // Init comms
    let universe = mpi::initialize().unwrap();
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

    local_blocks
        .attach_progress(block_count_progress)
        .for_each(|s| {
            local_block_counts
                .entry(s)
                .and_modify(|c|*c+=1)
                .or_insert(1);
        });
    

    if let Some(global_block_counts) = reduce_block_counts(&world, local_block_counts) {

        // Convert block_counts to blocks
        let block_tokenize_progress = Progress::new(
                Some(global_block_counts.len()), // length
                rank,
                "🔢 tokenizing blocks",
                Some("🔢 blocks tokenized")
            );

        let global_blocks: Vec<Block> = global_block_counts
            .into_iter()
            .attach_progress(block_tokenize_progress)
            .map(|(s, count)| {
                Block::new(s, count as i32)
            })
            .collect();
        
        // Init pair counter
        let bp_counts = PairCounter::new(&global_blocks,&world); 

        (global_blocks, bp_counts)
        
    } else {
        // Kill non root processes, freeing up
        // rescources for fast merging on root.
        // Don't actually need to kill the process: even if it is 
        // alive and idle the root can still use all resources.
        // However, this is the cleanest way I have found so far
        // (avoids handling ranks outside of this function)
        unsafe {mpi::ffi::MPI_Finalize()};
        std::process::exit(0)
    }
}