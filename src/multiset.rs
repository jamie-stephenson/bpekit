use crate::all_reduce_counts::all_reduce_counts;

use std::collections::HashMap;
use std::fmt;
use std::cmp::min;

use mpi::topology::SimpleCommunicator;

#[derive(Debug)]
struct Node {
    count: u64,
    val: (u32, u32),
    heap_pos: usize,
}

impl Node {
    fn new(count: u64, val: (u32, u32), heap_pos: usize) -> Self {
        Node { count, val, heap_pos }
    }
}

// struct to maintain counts across all processes
pub struct DistributedMultiset<'a> {
    nodes: Vec<Node>,                       // Array of nodes
    heap: Vec<usize>,                       // Heap of indices into nodes
    d: HashMap<(u32, u32), usize>,          // Map from value to node index
    to_add: HashMap<(u32, u32), u64>,       // Pending additions
    to_remove: HashMap<(u32, u32), u64>,    // Pending removals
    world: &'a SimpleCommunicator           // For collective comms
}

impl<'a> DistributedMultiset<'a> {

    pub fn new(data: Vec<Vec<u32>>, world: &'a SimpleCommunicator) -> Self {
        let mut ms = DistributedMultiset {
            nodes: Vec::new(),
            heap: Vec::new(),
            d: HashMap::new(),
            to_add: HashMap::new(),
            to_remove: HashMap::new(),
            world: world
        };

        // Process each sublist to extract pairs and add them to the Multiset
        for seq in data.iter() {
            // Extract adjacent pairs
            if seq.len() >= 2 {
                for pair in seq.windows(2) {
                    let a = pair[0];
                    let b = pair[1];
                    ms.add((a, b), 1);
                }
            }
        }

        // Commit pending additions
        ms.commit();

        ms
    }

    // Add an item with a count
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
        // `all_reduce_counts` has extra logic to ensure that all processes return the same 
        
        let to_add_local: Vec<((u32, u32), u64)> = self.to_add.drain().collect();
        let to_remove_local: Vec<((u32, u32), u64)> = self.to_remove.drain().collect();

        let to_add_global = all_reduce_counts(&self.world, to_add_local);
        let to_remove_global = all_reduce_counts(&self.world, to_remove_local);

        // Process additions
        for (item, count) in to_add_global {
            if !self.heap.is_empty() {
            }
            self.add_pair(item, count);
        }

        // Process removals
        for (item, count) in to_remove_global {
            self.remove_pair(item, count);
        }
    }

    // Internal method to add items
    fn add_pair(&mut self, item: (u32, u32), count: u64) {
        if count == 0 {
            return;
        }
        if let Some(&node_idx) = self.d.get(&item) {
            // To avoid multiple mutable borrows, extract heap_pos first
            let heap_pos = {
                let node = &mut self.nodes[node_idx];
                node.count += count;
                node.heap_pos
            };
            self.item_increased(heap_pos);
        } else {
            let node_idx = self.nodes.len();
            let heap_pos = self.heap.len();
            let node = Node::new(count, item, heap_pos);
            self.nodes.push(node);
            self.d.insert(item, node_idx);
            self.heap.push(node_idx);
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
                let node = &mut self.nodes[node_idx];

                
                node.count -= count;
                heap_pos = node.heap_pos;
            }
            self.item_decreased(heap_pos);
        }
    }

    // Get the count of an item
    pub fn count(&mut self, item: &(u32, u32)) -> u64 {
        self.commit();
        if let Some(&node_idx) = self.d.get(item) {
            self.nodes[node_idx].count
        } else {
            0
        }
    }

    // Get the most common item, and its count 
    pub fn most_common(&mut self) -> Option<((u32, u32),u64)> {
        self.commit();
        if !self.heap.is_empty() {
            let node_idx = self.heap[0];
            Some((self.nodes[node_idx].val,self.nodes[node_idx].count))
        } else {
            None
        }
    }

    // Check if the multiset is empty
    pub fn is_empty(&mut self) -> bool {
        self.commit();
        self.heap.is_empty() || self.nodes.iter().all(|node| node.count == 0)
    }

    // Maintain the heap when an item's count increases
    fn item_increased(&mut self, mut pos: usize) {
        let node_idx = self.heap[pos];
        while pos > 0 {
            let parent_pos = (pos - 1) / 2;
            let parent_idx = self.heap[parent_pos];
            // Compare counts directly
            if self.nodes[parent_idx].count < self.nodes[node_idx].count {
                // Swap positions
                self.heap[pos] = parent_idx;
                self.nodes[parent_idx].heap_pos = pos;
                pos = parent_pos;
            } else {
                break;
            }
        }
        self.heap[pos] = node_idx;
        self.nodes[node_idx].heap_pos = pos;
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

                if self.nodes[child_idx].count < self.nodes[right_child_idx].count {

                    child = right_child; // make sure `child` points to largest child
                    child_idx = right_child_idx;

                }
            }
            
            if self.nodes[node_idx].count < self.nodes[child_idx].count {
                // Swap positions
                self.heap[pos] = child_idx;
                self.nodes[child_idx].heap_pos = pos;
                pos = child;
                child = 2*pos + 1; // make sure `child` points to left child
            } else {
                break;
            }
        }
    
        self.heap[pos] = node_idx;
        self.nodes[node_idx].heap_pos = pos;
    }
    
}

impl<'a> fmt::Debug for DistributedMultiset<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.heap.is_empty() {
            writeln!(f, "Multiset is empty.")
        } else {
            self.print_heap(f)
        }
    }
}

impl<'a> DistributedMultiset<'a> {
    fn print_heap(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let heap_size = self.heap.len();
        let height = (heap_size as f64).log2().ceil() as usize;
        let max_width = 2usize.pow(height as u32) * 4; // Adjust width per node if needed

        for level in 0..height {
            let level_nodes = 2usize.pow(level as u32);
            let indent_space = max_width / (level_nodes * 2);
            let between_space = max_width / level_nodes - indent_space * 2;
            let start = level_nodes - 1;
            let end = min(start + level_nodes, heap_size);

            // Print nodes
            let mut line = String::new();
            for i in start..end {
                if i == start {
                    line.push_str(&" ".repeat(indent_space));
                } else {
                    line.push_str(&" ".repeat(between_space));
                }
                let node_idx = self.heap[i];
                let node = &self.nodes[node_idx];
                line.push_str(&format!("({},{})", node.val.0, node.val.1));
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

                    branch_line.push_str(&" ".repeat(3)); // Adjust if nodes have wider representation

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

#[cfg(test)]
mod tests {

    use super::*;
    use mpi::initialize;

    #[test]
    fn new_multiset_from_nested_vecs() {
        // Create a multiset from nested Vecs

        let universe = initialize().unwrap();
        let world = universe.world();

        let data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6, 7, 8, 1, 2],
            vec![1, 2],
            vec![7],
            vec![8, 1],
        ];

        let mut ms = DistributedMultiset::new(data,&world);

        // Check that counts for each pair are correct
        assert_eq!(ms.count(&(1, 2)), 3);  // (1,2) appears 3 times
        assert_eq!(ms.count(&(2, 3)), 1);  // (2,3) appears 1 time
        assert_eq!(ms.count(&(4, 5)), 1);  // (4,5) appears 1 time
        assert_eq!(ms.count(&(5, 6)), 1);  // (5,6) appears 1 time
        assert_eq!(ms.count(&(6, 7)), 1);  // (6,7) appears 1 time
        assert_eq!(ms.count(&(7, 8)), 1);  // (7,8) appears 1 time
        assert_eq!(ms.count(&(8, 1)), 2);  // (8,1) appears 2 times

        // Check that the most common pair is (1,2)
        assert_eq!(ms.most_common(), Some(((1, 2),3)));
    }

    #[test]
    fn add_and_remove_items() {

        let universe = initialize().unwrap();
        let world = universe.world();

        // Create a multiset from a simple nested Vec
        let data = vec![vec![1, 2, 3]];
        let mut ms = DistributedMultiset::new(data,&world);

        // Add more occurrences of the pair (2,3)
        ms.add((2, 3), 2); // Now (2,3) should have a count of 3
        assert_eq!(ms.count(&(2, 3)), 3);

        // Remove an occurrence of the pair (1,2)
        ms.remove((1, 2), 1); // Now (1,2) should have a count of 0
        assert_eq!(ms.count(&(1, 2)), 0);

        // Check that the most common pair is now (2,3)
        assert_eq!(ms.most_common(), Some(((2, 3),3)));
    }

    #[test]
    fn is_empty_after_removals() {

        let universe = initialize().unwrap();
        let world = universe.world();
        
        // Create a multiset with two sequential pairs
        let data = vec![vec![1, 2], vec![2, 3]];
        let mut ms = DistributedMultiset::new(data,&world);

        // Remove all pairs
        ms.remove((1, 2), 1);
        ms.remove((2, 3), 1);

        // The multiset should now be empty
        assert!(ms.is_empty());
    }

    #[test]
    fn large_dataset() {

        let universe = initialize().unwrap();
        let world = universe.world();
        
        // Create a large dataset where the same sequence repeats 1000 times
        let mut data = Vec::new();
        for _ in 0..1000 {
            data.push(vec![1, 2, 3, 4, 5]);
        }

        let mut ms = DistributedMultiset::new(data,&world);

        // Verify the counts of each pair
        assert_eq!(ms.count(&(1, 2)), 1000);  // (1,2) appears 1000 times
        assert_eq!(ms.count(&(2, 3)), 1000);  // (2,3) appears 1000 times
        assert_eq!(ms.count(&(3, 4)), 1000);  // (3,4) appears 1000 times
        assert_eq!(ms.count(&(4, 5)), 1000);  // (4,5) appears 1000 times
    }

    #[test]
    fn zero_count() {

        let universe = initialize().unwrap();
        let world = universe.world();

        // Create a multiset with a few pairs
        let data = vec![vec![1, 2, 3]];
        let mut ms = DistributedMultiset::new(data,&world);

        // Remove an occurrence of (1,2)
        ms.remove((1, 2), 1);

        // (1,2) should now have a count of 0
        assert_eq!(ms.count(&(1, 2)), 0);

        // Check that the multiset is not empty (since (2,3) still exists)
        assert!(!ms.is_empty());
    }

    #[test]
    fn removing_nonexistent_item() {

        let universe = initialize().unwrap();
        let world = universe.world();

        // Create a multiset with a few pairs
        let data = vec![vec![1, 2, 3]];
        let mut ms = DistributedMultiset::new(data,&world);

        // Try removing a nonexistent pair (3,4) (it doesn't exist)
        ms.remove((3, 4), 1);

        // The count of (3,4) should still be 0
        assert_eq!(ms.count(&(3, 4)), 0);

        // The multiset should still contain other valid pairs
        assert_eq!(ms.count(&(1, 2)), 1);
        assert_eq!(ms.count(&(2, 3)), 1);
    }

    #[test]
    fn most_common_with_ties() {

        let universe = initialize().unwrap();
        let world = universe.world();

        // Create a multiset where several pairs have the same count
        let data = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let mut ms = DistributedMultiset::new(data,&world);

        // All pairs have a count of 1, so any of them could be the most common
        let (most_common, _count) = ms.most_common().unwrap();
        assert!(most_common == (1, 2) || most_common == (3, 4) || most_common == (5, 6));
    }

    #[test]
        fn large_counts() {


        let universe = initialize().unwrap();
        let world = universe.world();

        // Create a multiset with a few pairs
        let data = vec![vec![1, 2, 3]];
        let mut ms = DistributedMultiset::new(data,&world);

        // Add a large count to (1,2)
        ms.add((1, 2), 1_000_000);

        // Verify that the count for (1,2) is now 1_000_001
        assert_eq!(ms.count(&(1, 2)), 1_000_001);

        // The most common pair should be (1,2)
        assert_eq!(ms.most_common(), Some(((1, 2),1_000_001)));
    }
}

