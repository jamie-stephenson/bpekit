import os

def get_rank_and_world_size():
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK',0))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE',1))
    return rank, world_size