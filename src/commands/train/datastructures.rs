// Structures for maintaining pair counts during training

use super::comms::all_reduce_changes;
use crate::utils::progress::get_progress_reporter;

use std::collections::{HashMap,HashSet,BinaryHeap};
use std::cmp::Ordering;

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
pub(crate) struct PairCounter<'a> {
    heap: BinaryHeap<Pair>,                                 // Heap of pairs, maintained based on global counts
    counts: HashMap<(u32, u32), i32>,                       // Map from pair to global count, used to validate the top of the heap
    world: &'a SimpleCommunicator                           // For collective comms
}

impl<'a> PairCounter<'a> {

    pub fn new(blocks: &[Block], world: &'a SimpleCommunicator) -> Self {
        let mut pc = PairCounter {
            heap: BinaryHeap::new(),
            counts: HashMap::new(),
            world: world
        };

        let mut counts_map: HashMap<(u32,u32),(i32,HashSet<usize>)> = HashMap::new();
        
        let mut progress = get_progress_reporter(
            Some(blocks.len()),
            world.rank(),
            "🧮 counting pairs", 
            None,
            1000000, 
            true
        );

        // Process each block to extract pairs and add them to the heap
        for (block_idx,block) in blocks.into_iter().enumerate() {

            // Extract adjacent pairs
            if block.tokens.len() >= 2 {
                for pair in block.tokens.windows(2) {
                    let a = pair[0] as u32;
                    let b = pair[1] as u32;
                    counts_map.entry((a, b))
                        .and_modify(|(count, set)| {
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
            progress.inc(1);
        }
        
        // Convert to Vec
        let counts_vec: Vec<((u32, u32), (i32, Vec<usize>))> = counts_map
        .into_iter()
        .map(|(pair, (count, block_ids))| (pair, (count, block_ids.into_iter().collect())))
        .collect();
    
        progress.finish_with_message("🧮 pairs counted");

        // Commit pending changes
        pc.commit(counts_vec);
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
    pub fn commit(&mut self, changes: Vec<((u32,u32),(i32,Vec<usize>))>) {

        let (local_changes, local_new_block_ids): (Vec<((u32,u32),i32)>, HashMap<(u32,u32),Vec<usize>>) = changes
            .into_iter()
            .map(|(pair, (change, block_ids))| ((pair, change), (pair, block_ids)))
            .unzip();

        let global_changes = all_reduce_changes(&self.world, local_changes); 

        // Process changes
        for (pair, change) in global_changes {
            
            // You only ever have a positive change when you are introducing a new pair.
            // Add new pairs to the heap. 
            if change > 0 { 
                self.counts.insert(pair, change);

                let block_ids = local_new_block_ids
                    .get(&pair)
                    .cloned()
                    .unwrap_or_else(Vec::new);

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