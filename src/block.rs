use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

// A string with n utf-8 characters will begin the bpe process by being represented by
// n `Token`s, grouped together into `Block`s. Every iteration `Token`s will be "merged".
#[derive(Debug)]
pub(super) struct Token {
    pub val: u32,
    pub prev: Option<usize>,
    pub next: Option<usize>,
}

impl Token {
    pub fn merge_with(&mut self, next_token: &Self, new_val: u32) {
        self.val = new_val;
        self.next = next_token.next;
    }   
}

// Usually represents a word with possible preceding space but this can vary 
// depending on the regex pattern used to split the text. 
pub(super) struct Block {
    pub tokens: Vec<Token>
}

impl Block {
    pub(super) fn new() -> Self {
        Block {
            tokens: vec![]
        }
    }

    pub(super) fn add(&mut self, val: u32) {
        let (prev, next) = {
            let len = self.tokens.len() as usize;
            if let Some(prev_token) = self.tokens.last_mut() {
                prev_token.next = Some(len);
                (Some(len - 1), None)
            } else {
                (None, None)
            }
        };
        self.tokens.push(
            Token { 
                val: val, 
                prev: prev, 
                next: next 
            }
        )
    }
}


// Define the IndexedBlocks structure
pub(crate) struct IndexedBlocks {
    pub index: HashMap<(u32, u32), Vec<Weak<RefCell<Node>>>>,
    pub blocks: Vec<IndexedList>,
}

impl IndexedBlocks {
    pub fn new(blocks: Vec<Vec<u32>>) -> Self {
        let mut indexed_blocks = IndexedBlocks {
            blocks: Vec::new(),
            index: HashMap::new(),
        };

        // For each block, create a IndexedList and merge its index
        for block in blocks {
            let mut indexed_list = IndexedList::new(block);
            let index = std::mem::take(&mut indexed_list.index);

            // Move the current list's index and merge it into the global index
            for (pair, nodes) in index {
                indexed_blocks
                    .index
                    .entry(pair)
                    .or_insert_with(Vec::new)
                    .extend(nodes);
            }

            // Add the indexed list to the blocks
            indexed_blocks.blocks.push(indexed_list);
        }

        indexed_blocks
    }
    
    pub fn update_index(&mut self, node: &Rc<RefCell<Node>>) {
        let node_borrowed = node.borrow();
        if let Some(prev) = &node_borrowed.prev {
            if let Some(prev_rc) = prev.upgrade() {
                self.add_to_index(
                    (prev_rc.borrow().val, node_borrowed.val),
                    &prev_rc,
                );
            }
        }
        if let Some(next) = &node_borrowed.next {
            self.add_to_index(
                (node_borrowed.val, next.borrow().val),
                node,
            );
        }
    }

    pub fn drain_blocks(self) -> Vec<Vec<u32>> {
        let mut blocks_vec = Vec::new();

        for mut indexed_list in self.blocks {
            let mut block = Vec::new();
            let mut current_node = indexed_list.head.take();  
            // Traverse the linked list, consuming nodes as we go
            while let Some(node_rc) = current_node {
                let mut node = node_rc.borrow_mut();
                block.push(node.val);  // Add the value to the current block

                current_node = node.next.take();
            }

            blocks_vec.push(block);
        }

        blocks_vec
    }

    fn add_to_index(&mut self, pair: (u32, u32), node: &Rc<RefCell<Node>>) {
        self.index
            .entry(pair)
            .or_insert_with(Vec::new)
            .push(Rc::downgrade(node));
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_rust_node_creation() {
        let node = Node::new(5);
        assert_eq!(node.borrow().val, 5);
        assert!(node.borrow().prev.is_none());
        assert!(node.borrow().next.is_none());
    }

    #[test]
    fn test_rust_node_linking() {
        let node1 = Node::new(1);
        let node2 = Node::new(2);

        // Link node1 and node2
        node1.borrow_mut().next = Some(node2.clone());
        node2.borrow_mut().prev = Some(Rc::downgrade(&node1));

        assert!(node1.borrow().next.is_some());
        assert!(node2.borrow().prev.is_some());
        assert_eq!(node1.borrow().next.as_ref().unwrap().borrow().val, 2);
        assert_eq!(node2.borrow().prev.as_ref().unwrap().upgrade().unwrap().borrow().val, 1);
    }

    #[test]
    fn test_rust_node_deletion() {
        let node1 = Node::new(1);
        let node2 = Node::new(2);
        let node3 = Node::new(3);

        // Link the nodes
        node1.borrow_mut().next = Some(node2.clone());
        node2.borrow_mut().prev = Some(Rc::downgrade(&node1));
        node2.borrow_mut().next = Some(node3.clone());
        node3.borrow_mut().prev = Some(Rc::downgrade(&node2));

        // Delete node2
        node2.borrow_mut().delete();

        // Ensure node1's next is node3
        assert_eq!(node1.borrow().next.as_ref().unwrap().borrow().val, 3);
        // Ensure node3's prev is node1
        assert_eq!(node3.borrow().prev.as_ref().unwrap().upgrade().unwrap().borrow().val, 1);
    }

    #[test]
    fn test_rust_indexed_list_creation() {
        let vals = vec![1, 2, 3, 4];
        let indexed_list = IndexedList::new(vals);

        assert_eq!(indexed_list.index.len(), 3); // Expect (1, 2), (2, 3), (3, 4)
        assert!(indexed_list.index.contains_key(&(1, 2)));
        assert!(indexed_list.index.contains_key(&(2, 3)));
        assert!(indexed_list.index.contains_key(&(3, 4)));
    }

    #[test]
    fn test_rust_indexed_blocks_creation() {
        let blocks = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let indexed_blocks = IndexedBlocks::new(blocks);

        // Check that all the pairs across blocks are indexed
        assert_eq!(indexed_blocks.index.len(), 4); // Expect (1, 2), (2, 3), (4, 5), (5, 6)
        assert!(indexed_blocks.index.contains_key(&(1, 2)));
        assert!(indexed_blocks.index.contains_key(&(2, 3)));
        assert!(indexed_blocks.index.contains_key(&(4, 5)));
        assert!(indexed_blocks.index.contains_key(&(5, 6)));
    }

    #[test]
    fn test_rust_indexed_blocks_update_index() {
        let blocks = vec![vec![1, 2, 3]];
        let mut indexed_blocks = IndexedBlocks::new(blocks);

        // Create a new node and link it into the list
        let node1 = indexed_blocks.index[&(1, 2)][0].upgrade().unwrap();
        let node4 = Node::new(4);
        node1.borrow_mut().next = Some(node4.clone());
        node4.borrow_mut().prev = Some(Rc::downgrade(&node1));

        // Update the index to reflect the change
        indexed_blocks.update_index(&node4);

        // Check that the new pair (1, 4) is in the index
        assert!(indexed_blocks.index.contains_key(&(1, 4)));
    }

    #[test]
    fn test_rust_indexed_blocks_update_node_value() {
        let blocks = vec![vec![1, 2, 3]];
        let mut indexed_blocks = IndexedBlocks::new(blocks);

        // Retrieve node1 and node2 using the global index
        let node1 = indexed_blocks.index[&(1, 2)][0].upgrade().unwrap();
        let node2 = node1.borrow().next.clone().unwrap();
        let node3 = node2.borrow().next.clone().unwrap();

        // Change the value of node2 from 2 to 4
        node2.borrow_mut().val = 4;

        // Call update_index on node2
        indexed_blocks.update_index(&node2);

        // Verify that the list is now 1 -> 4 -> 3
        assert_eq!(node1.borrow().val, 1);
        assert_eq!(node1.borrow().next.as_ref().unwrap().borrow().val, 4);
        assert_eq!(
            node1
                .borrow()
                .next
                .as_ref()
                .unwrap()
                .borrow()
                .next
                .as_ref()
                .unwrap()
                .borrow()
                .val,
            3
        );

        // Also verify that 1 <- 4 <- 3
        assert_eq!(node3.borrow().val, 3);
        assert_eq!(
            node3
                .borrow()
                .prev
                .as_ref()
                .unwrap()
                .upgrade()
                .unwrap()
                .borrow()
                .val,
            4
        );
        assert_eq!(
            node3
                .borrow()
                .prev
                .as_ref()
                .unwrap()
                .upgrade()
                .unwrap()
                .borrow()
                .prev
                .as_ref()
                .unwrap()
                .upgrade()
                .unwrap()
                .borrow()
                .val,
            1
        );

        // Check that the new pairs are in the index
        assert!(indexed_blocks.index.contains_key(&(1, 4)));
        assert!(indexed_blocks.index.contains_key(&(4, 3)));
    }

    #[test]
    fn test_drain_blocks() {
        // Create some blocks of integers
        let blocks = vec![
            vec![1, 2, 3],
            vec![4, 5],
            vec![6, 7, 8, 9],
        ];

        let indexed_blocks = IndexedBlocks::new(blocks.clone());

        let result = indexed_blocks.drain_blocks();

        assert_eq!(result, blocks);
    }
}