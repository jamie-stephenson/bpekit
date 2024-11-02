//  Datastructures for use during encoding
use std::cmp::Ordering;

// Each `Merge` corresponds to a single instance of a consective pair of tokens during the
// encoding process.
#[derive(Debug, Eq)]
pub struct Merge {
    pub idx: usize,  // Index of pair within object currently being encoded
    pub val: u32     // New token to merge pair into.
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val && self.idx == other.idx
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// This helps define a min heap
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.val != other.val {
            other.val.cmp(&self.val)
        } else {
            other.idx.cmp(&self.idx)
        }
    }
}

#[derive(Clone, Copy)]
pub struct Token {
    pub val: u32,
    pub prev: Option<usize>,
    pub next: Option<usize>,
    pub width: usize // Number of utf-8 tokens merged to create this token
}

impl Token {
    pub fn merge(&mut self, next_token: &Self, new: u32) {
        self.val = new;
        self.next = next_token.next;
        self.width += next_token.width;
    }
}