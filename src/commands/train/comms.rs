use std::collections::HashMap;

use bincode::{serialize,deserialize};
use mpi::traits::*;
use mpi::topology::{Communicator,SimpleCommunicator};
use mpi::datatype::PartitionMut;
use mpi::Count;


pub(crate) fn reduce_block_counts(
    world: &SimpleCommunicator,
    local_map: HashMap<String, i32>,
) -> Option<HashMap<String, i32>> {

    let serialized_bytes = match serialize(&local_map) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Serialization error: {}", e);
            return None;
        }
    };

    if let Some(gathered_bytes) = gather_bytes(world, serialized_bytes) {

        let mut aggregated_map = HashMap::new();

        for bytes in gathered_bytes {

            let received_map: HashMap<String, i32> = match deserialize(&bytes) {
                Ok(map) => map,
                Err(e) => {
                    eprintln!("Deserialization error: {}", e);
                    continue;
                }
            };
            for (key, value) in received_map {
                *aggregated_map.entry(key).or_insert(0) += value;
            }
        }

        Some(aggregated_map)

    } else {
        None
    }
}


fn gather_bytes(
    world: &mpi::topology::SimpleCommunicator,
    bytes: Vec<u8>,
) -> Option<Vec<Vec<u8>>> {
    let rank = world.rank();
    let root = world.process_at_rank(0);

    // Determine the size of the serialized data
    let local_size: Count = bytes.len() as Count;

    // Gather all sizes to the root process
    let mut sizes = if rank == 0 {
        vec![0; world.size() as usize]
    } else {
        vec![]
    };

    match rank {
        0 => root.gather_into_root(&local_size, &mut sizes),
        _ => root.gather_into(&local_size)
    }

    if rank == 0 {
        // Calculate displacements based on sizes
        let displs: Vec<Count> = sizes
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect();

        // Allocate a buffer to receive all serialized data
        let global_size: Count = sizes.iter().sum();
        let mut recv_buffer = vec![0u8; global_size as usize];

        // Create a PartitionMut to specify where each process's data will be placed
        let mut partition = PartitionMut::new(
            &mut recv_buffer[..],
            &sizes[..],
            &displs[..]
        );

        // Gather all serialized bytes into the receive buffer
        root.gather_varcount_into_root(&bytes[..], &mut partition);

        let mut result: Vec<Vec<u8>> = Vec::with_capacity(world.size() as usize);

        for i in 0..world.size() {
            let size = sizes[i as usize] as usize;
            if size == 0 {
                continue; // Skip if the process sent no data
            }
            let start = displs[i as usize] as usize;
            let end = start + size;
            result.push(recv_buffer[start..end].to_vec());
        }
        Some(result)
    } else {
        // Non-root processes send their serialized bytes
        root.gather_varcount_into(&bytes[..]);
        None
    }
}
