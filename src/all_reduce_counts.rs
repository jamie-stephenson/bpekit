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
pub fn all_reduce_counts(
    world: &mpi::topology::SimpleCommunicator,
    local_counts: Vec<((u32, u32), u64)>,
) -> Vec<((u32, u32), u64)> {

    let local_size = local_counts.len() * 2;
    let mut flat_local_data = Vec::with_capacity(local_size);

    // Flatten local counts into a vector of u64s.
    for &((x, y), count) in &local_counts {
        let key = combine_u32(x, y);
        flat_local_data.push(key);
        flat_local_data.push(count);
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

    // Combine counts with the same key
    let mut map: HashMap<u64, u64> = HashMap::new();
    // Vector to store the order of insertion
    let mut order: Vec<u64> = Vec::new();

    for chunk in buf.chunks_exact(2) {
        let key = chunk[0];
        let count = chunk[1];
        if let Some(entry) = map.get_mut(&key) {
            // If the key exists, add the count
            *entry += count;
        } else {
            // If the key doesn't exist, insert it and remember the insertion order
            map.insert(key, count);
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
    fn test_empty_vec() {
        let universe = initialize().unwrap();
        let world = universe.world();

        // Each process starts with an empty vector.
        let local_counts: Vec<((u32, u32), u64)> = vec![];

        // Perform the all-reduce operation.
        let global_counts = all_reduce_counts(&world, local_counts);

        // Since we started with an empty vector, the result should also be empty.
        assert!(global_counts.is_empty(), "Expected empty global_counts, got {:?}", global_counts);
    }

    #[test]
    fn test_single_process() {
        let universe = initialize().unwrap();
        let world = universe.world();

        if world.size() == 1 {
            // Each process creates its local_counts vector.
            let local_counts = vec![
                ((0, 1), 5),
                ((1, 2), 10),
                ((2, 3), 15),
            ];

            // Perform the all-reduce operation.
            let global_counts = all_reduce_counts(&world, local_counts.clone());

            // Since we have only one process, the global counts should be identical to local_counts.
            assert_eq!(global_counts, local_counts, "Expected {:?}, got {:?}", local_counts, global_counts);
        }
    }

    #[test]
    fn test_multiple_processes_unique_tuples() {
        let universe = initialize().unwrap();
        let world = universe.world();

        // Each process will generate unique tuples based on its rank.
        let rank = world.rank() as u32;
        let local_counts = vec![
            ((rank, rank + 1), 1),
            ((rank + 1, rank + 2), 2),
        ];

        // Perform the all-reduce operation.
        let global_counts = all_reduce_counts(&world, local_counts);

        // Gather the expected results based on the number of processes.
        let mut expected_counts = vec![];
        for rank in 0..world.size() as u32 {
            expected_counts.push(((rank, rank + 1), 1));
            expected_counts.push(((rank + 1, rank + 2), 2));
        }

        // Sort the vectors for easier comparison (since order may vary).
        let mut global_counts_sorted = global_counts.clone();
        global_counts_sorted.sort_by_key(|&((a, b), _)| (a, b));
        expected_counts.sort_by_key(|&((a, b), _)| (a, b));

        // Assert that the global counts match the expected counts.
        assert_eq!(global_counts_sorted, expected_counts, "Expected {:?}, got {:?}", expected_counts, global_counts_sorted);
    }

    #[test]
    fn test_multiple_processes_shared_tuples() {
        let universe = initialize().unwrap();
        let world = universe.world();

        // Each process will create tuples with the same keys but different counts.
        let rank = world.rank() as u32;
        let local_counts = vec![
            ((0, 1), rank as u64 + 1),
            ((1, 2), (rank as u64 + 1) * 2),
        ];

        // Perform the all-reduce operation.
        let global_counts = all_reduce_counts(&world, local_counts);

        // Calculate the expected sum of counts across all processes.
        let size = world.size() as u64;
        let expected_counts = vec![
            ((0, 1), (size * (size + 1)) / 2),
            ((1, 2), (size * (size + 1))),
        ];

        // Sort the vectors for easier comparison.
        let mut global_counts_sorted = global_counts.clone();
        global_counts_sorted.sort_by_key(|&((a, b), _)| (a, b));

        // Assert that the global counts match the expected counts.
        assert_eq!(global_counts_sorted, expected_counts, "Expected {:?}, got {:?}", expected_counts, global_counts_sorted);
    }
}
