mod datastructures;
mod save;

use datastructures::{Merge,Token};
use save::save_tokens;
use std::collections::{BinaryHeap,HashMap};
use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyString};
use pyo3::exceptions::PyException;


/// Encode a dataset and save as numpy shards. 
#[pyfunction]
pub fn encode_dataset(
    generator: &Bound<'_,PyIterator>, 
    merges: Vec<((u32,u32),u32)>,
    path: &str,
    shard_size: usize,
    rank: i32
) -> PyResult<()> {

    let mut merge_map: HashMap<(u32,u32), u32> = HashMap::new();

    for (pair,token) in merges {
        merge_map.insert(pair, token);
    }

    let tokens = generator
        .into_iter()    
        .map(|s| {
            // TODO: Handle Result better, don't just unwrap
            let string = s.unwrap()
                .downcast::<PyString>().unwrap()
                .to_string();
            encode_string(string, &merge_map)
        });
    
    save_tokens(tokens, Path::new(path), shard_size, rank)
        .map_err(|e| PyException::new_err(e.to_string()))
}

/// Encode a string to integer tokens
#[pyfunction]
pub fn encode(s: String, merge_map: HashMap<(u32,u32),u32>) -> Vec<u32> {
    encode_string(s, &merge_map)
}

fn encode_string(s: String, merge_map: &HashMap<(u32,u32),u32>) -> Vec<u32> {

    let utf8 = s.as_bytes().to_vec();
    let length = utf8.len(); 
    let mut queue = BinaryHeap::with_capacity(length);
    
    queue.extend(utf8
        .windows(2)
        .enumerate()
        .filter_map(|(idx, window)| {
            let pair = (window[0] as u32, window[1] as u32);
            merge_map.get(&pair).map(|val| Merge{idx,val:*val})
        }),
    );

    let mut tokens: Vec<Token> = utf8
        .into_iter()
        .enumerate()
        .map(|(i, byte)| {
            Token {
                val: byte as u32,
                prev: if i > 0 { Some(i - 1) } else { None },
                next: if i + 1 < length { Some(i + 1) } else { None },
                width: 1,
            }
        }).collect();
    
    while let Some(merge) = queue.pop() {
        // Each merge looks like this: 
        //          `merge.idx` points here
        //                       VVVVVVVV    
        //      ... prev_token, left_token, right_token, next_token ...
        //                      \__We merge these two__/
        //
        // It is possible that some of these do not exist or the merge is stale so we need to check.
        
        // ---- Check if right_token exists ----
        let right_idx = match tokens[merge.idx].next {
            Some(item) => item,
            None => continue
        };
        let right_token = tokens[right_idx];
        
        // ---- Check if merge is stale ----
        // Scenario 1: "swallowed"
        // e.g. tokens: abc, merges ab, bc
        // abc -> X_c but the merge bc still indexes "_" even though it has been
        // "swallowed" by the previous merge. We lazily check this just before attempting
        // to merge:
        if tokens[merge.idx].width == 0 {
            continue;
        }
        
        // Scenario 2: "new pointer"
        // e.g. tokens: abc, merges bc, ab
        // abc -> aX_ but the merge ab still indexes "a" even though a no longer points
        // to b. We also lazily check this.
        let pair = (tokens[merge.idx].val,right_token.val);
        if !merge_map.get(&pair).map_or(false, |new_token| *new_token == merge.val ) {
            continue;
        }
        
        // ---- Perform merge ----
        tokens[merge.idx].merge(&right_token, merge.val);
        tokens[right_idx].width = 0; // right token is "swallowed"
        
        let left_token = tokens[merge.idx];

        // Handle results of merge looking forward
        if let Some(next_idx) = left_token.next {

            tokens[next_idx].prev = Some(merge.idx);

            let new_pair = (left_token.val,tokens[next_idx].val);
            if let Some(val) = merge_map.get(&new_pair) {
                queue.push(Merge { idx: merge.idx, val: *val })
            }
        }

        // Handle the results of the merge looking backward
        if let Some(prev_idx) = left_token.prev {

            let new_pair = (tokens[prev_idx].val,left_token.val);
            if let Some(val) = merge_map.get(&new_pair) {
                queue.push(Merge { idx: prev_idx, val: *val })
            }
        }
    };
    tokens
        .into_iter()
        .filter_map(|token| (token.width != 0).then(|| token.val))
        .collect()
}