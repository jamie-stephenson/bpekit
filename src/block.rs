use std::collections::HashMap;

pub(crate) struct Block {
    pub tokens: Vec<u32>,
    pub count: i32
}

impl Block {
    pub fn new(utf8_codepoints: Vec<u8>, count: i32) -> Block {
        Block{ 
            tokens: utf8_codepoints
                .into_iter()
                .map(|byte| byte as u32)
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