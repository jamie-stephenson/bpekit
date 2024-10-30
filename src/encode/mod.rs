mod datastructures;

use datastructures::{Merge,Token};

use std::collections::{BinaryHeap,HashMap};

use pyo3::pyfunction;


#[pyfunction]
pub fn encode(utf8_codepoints: Vec<u8>, merges: HashMap<(u32,u32),u32>) -> Vec<u32> {

    let length = utf8_codepoints.len(); 
    let mut queue = BinaryHeap::with_capacity(length);
    
    queue.extend(utf8_codepoints
        .windows(2)
        .enumerate()
        .filter_map(|(idx, window)| {
            let pair = (window[0] as u32, window[1] as u32);
            merges.get(&pair).map(|val| Merge{idx,val:*val})
        }),
    );

    
    let mut tokens: Vec<Token> = utf8_codepoints
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
        if !merges.get(&pair).map_or(false, |new_token| *new_token == merge.val ) {
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
            if let Some(val) = merges.get(&new_pair) {
                queue.push(Merge { idx: merge.idx, val: *val })
            }
        }

        // Handle the results of the merge looking backward
        if let Some(prev_idx) = left_token.prev {

            let new_pair = (tokens[prev_idx].val,left_token.val);
            if let Some(val) = merges.get(&new_pair) {
                queue.push(Merge { idx: prev_idx, val: *val })
            }
        }
    };
    tokens
        .into_iter()
        .filter_map(|token| (token.width != 0).then(|| token.val))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_no_merges() {
        let utf8_codepoints = vec![1u8, 2, 3];
        let merges = HashMap::new();
        let result = encode(utf8_codepoints.clone(), merges);
        let expected: Vec<u32> = utf8_codepoints.into_iter().map(|b| b as u32).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_single_merge() {
        let utf8_codepoints = vec![1u8, 2, 3];
        let mut merges = HashMap::new();
        merges.insert((1u32, 2u32), 4u32);
        let result = encode(utf8_codepoints, merges);
        let expected = vec![4u32, 3u32];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiple_merges() {
        let utf8_codepoints = vec![1u8, 2, 3, 4];
        let mut merges = HashMap::new();
        merges.insert((1u32, 2u32), 5u32);
        merges.insert((3u32, 4u32), 6u32);
        merges.insert((5u32, 6u32), 7u32);
        let result = encode(utf8_codepoints, merges);
        let expected = vec![7u32];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_conflicting_merges() {
        let utf8_codepoints = vec![1u8, 2, 3];
        let mut merges = HashMap::new();
        merges.insert((1u32, 2u32), 4u32);
        merges.insert((2u32, 3u32), 5u32);
        let result = encode(utf8_codepoints, merges);
        let expected = vec![4u32, 3u32];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_stale_merges() {
        let utf8_codepoints = vec![1u8, 2, 1, 2];
        let mut merges = HashMap::new();
        merges.insert((1u32, 2u32), 4u32);
        merges.insert((2u32, 1u32), 3u32);
        let result = encode(utf8_codepoints, merges);
        let expected = vec![1u32, 3u32, 2u32];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_empty_input() {
        let utf8_codepoints = vec![];
        let merges = HashMap::new();
        let result = encode(utf8_codepoints, merges);
        let expected: Vec<u32> = vec![];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_single_codepoint() {
        let utf8_codepoints = vec![1u8];
        let merges = HashMap::new();
        let result = encode(utf8_codepoints, merges);
        let expected = vec![1u32];
        assert_eq!(result, expected);
    }
}
