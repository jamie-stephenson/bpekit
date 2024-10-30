use std::collections::HashMap;

use mpi::traits::*;
use mpi::topology::Communicator;
use mpi::datatype::PartitionMut;
use mpi::Count;

/// Combines two u32 values into a single u64 key.
fn combine_u32(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

/// Splits a u64 key back into two u32 values.
fn split_u64(key: u64) -> (u32, u32) {
    let a = (key >> 32) as u32;
    let b = key as u32;
    (a, b)
}

// All Reduce collective communication that ensures the same output on all ranks
pub fn all_reduce_changes(
    world: &mpi::topology::SimpleCommunicator,
    local_changes: Vec<((u32, u32), i32)>,
) -> Vec<((u32, u32), i32)> {
    
    let local_size = local_changes.len() * 2;
    let mut flat_local_data = Vec::with_capacity(local_size);

    // Flatten local changes into a vector of u64s.
    // We will never interpret these numbers as u64s,
    // this is just my first hacky mpi attempt and
    // 64 seemed like the best bitwidth to use to combine
    // stuff into.
    for &((x, y), change) in &local_changes {
        let key = combine_u32(x, y);
        flat_local_data.push(key);
        flat_local_data.push(change as u64);
    }


    let local_size_as_count = local_size as Count;
    let mut all_data_sizes = vec![0 as Count; world.size() as usize]; 
    world.all_gather_into(&local_size_as_count, &mut all_data_sizes);
    
    let displs: Vec<Count> = all_data_sizes
    .iter()
    .scan(0, |acc, &x| {
        let tmp = *acc;
        *acc += x;
        Some(tmp)
    })
    .collect();

    let global_size: i32 = all_data_sizes.iter().sum();
    let mut buf = vec![0; global_size as usize];    
    {
        let mut partition = PartitionMut::new(
            &mut buf[..], 
            all_data_sizes, 
            &displs[..]);
        world.all_gather_varcount_into(&flat_local_data[..], &mut partition);
    }

    // Combine changes with the same key
    let mut map: HashMap<u64, i32> = HashMap::new();
    // Vector to store the order of insertion
    let mut order: Vec<u64> = Vec::new();

    for chunk in buf.chunks_exact(2) {
        let key = chunk[0];
        let change = chunk[1] as i32;

        if let Some(entry) = map.get_mut(&key) {
            // If the key exists, add the count
            *entry += change;
        } else {
            // If the key doesn't exist, insert it and remember the insertion order
            map.insert(key, change);
            order.push(key);
        }
    }

    order.into_iter()
        .map(|key| (split_u64(key), map[&key]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpi::initialize;

    #[test]
    fn test_combine_u32() {
        let a: u32 = 12345;
        let b: u32 = 67890;
        let combined = combine_u32(a, b);
        assert_eq!(combined, 53021371337010);
    }

    #[test]
    fn test_split_u64() {
        let combined: u64 = 53021371337010;
        let (a, b) = split_u64(combined);
        assert_eq!(a, 12345);
        assert_eq!(b, 67890);
    }

    #[test]
    fn test_combine_and_split() {
        let a: u32 = 98765;
        let b: u32 = 43210;
        let combined = combine_u32(a, b);
        let (a_split, b_split) = split_u64(combined);
        assert_eq!(a, a_split);
        assert_eq!(b, b_split);
    }

    #[test]
    fn test_empty_vec() {
        let universe = initialize().unwrap();
        let world = universe.world();

        // Each process starts with an empty vector.
        let local_changes: Vec<((u32, u32), i32)> = vec![];

        // Perform the all-reduce operation.
        let global_changes = all_reduce_changes(&world, local_changes);

        // Since we started with an empty vector, the result should also be empty.
        assert!(global_changes.is_empty(), "Expected empty global_changes, got {:?}", global_changes);
    }

    #[test]
    fn test_single_process() {
        let universe = initialize().unwrap();
        let world = universe.world();

        if world.size() == 1 {
            // Each process creates its local_changes vector.
            let local_changes = vec![
                ((0, 1), 5),
                ((1, 2), 10),
                ((2, 3), 15),
            ];

            // Perform the all-reduce operation.
            let global_changes = all_reduce_changes(&world, local_changes.clone());

            // Since we have only one process, the global changes should be identical to local_changes.
            assert_eq!(global_changes, local_changes, "Expected {:?}, got {:?}", local_changes, global_changes);
        }
    }

    #[test]
    fn test_multiple_processes_unique_tuples() {
        let universe = initialize().unwrap();
        let world = universe.world();

        // Each process will generate unique tuples based on its rank.
        let rank = world.rank() as u32;
        let local_changes = vec![
            ((rank, rank + 1), 1),
            ((rank + 1, rank + 2), 2),
        ];

        // Perform the all-reduce operation.
        let global_changes = all_reduce_changes(&world, local_changes);

        // Gather the expected results based on the number of processes.
        let mut expected_changes = vec![];
        for rank in 0..world.size() as u32 {
            expected_changes.push(((rank, rank + 1), 1));
            expected_changes.push(((rank + 1, rank + 2), 2));
        }

        // Sort the vectors for easier comparison (since order may vary).
        let mut global_changes_sorted = global_changes.clone();
        global_changes_sorted.sort_by_key(|&((a, b), _)| (a, b));
        expected_changes.sort_by_key(|&((a, b), _)| (a, b));

        // Assert that the global changes match the expected changes.
        assert_eq!(global_changes_sorted, expected_changes, "Expected {:?}, got {:?}", expected_changes, global_changes_sorted);
    }

    #[test]
    fn test_multiple_processes_shared_tuples() {
        let universe = initialize().unwrap();
        let world = universe.world();

        // Each process will create tuples with the same keys but different changes.
        let rank = world.rank() as u32;
        let local_changes = vec![
            ((0, 1), rank as i32 + 1),
            ((1, 2), (rank as i32 + 1) * 2),
        ];

        // Perform the all-reduce operation.
        let global_changes = all_reduce_changes(&world, local_changes);

        // Calculate the expected sum of changes across all processes.
        let size = world.size() as i32;
        let expected_changes = vec![
            ((0, 1), (size * (size + 1)) / 2),
            ((1, 2), (size * (size + 1))),
        ];

        // Sort the vectors for easier comparison.
        let mut global_changes_sorted = global_changes.clone();
        global_changes_sorted.sort_by_key(|&((a, b), _)| (a, b));

        // Assert that the global changes match the expected changes.
        assert_eq!(global_changes_sorted, expected_changes, "Expected {:?}, got {:?}", expected_changes, global_changes_sorted);
    }
}
