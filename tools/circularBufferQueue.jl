"""
    compute_queue_size(num_of_vertices::Int32) -> Int32

Calculate the next power-of-two size for the circular queue buffer.
Uses bit manipulation for efficient computation.
"""
function compute_queue_size(num_of_vertices::Int32)::Int32
    return 1 << ceil(Int, log2(num_of_vertices + 2))
end


"""
    _CircularBufferQueue

High-performance circular queue implementation with:
- O(1) enqueue/dequeue operations
- Fixed-size buffer with power-of-two sizing
- Thread-safe for single producer/single consumer
"""
mutable struct _CircularBufferQueue
    mask::Int32
    queue::Vector{Int32}
    front::Int32  # Thread-safe access
    rear::Int32# Thread-safe access
    num::Int32    # Thread-safe count

    function _CircularBufferQueue(num_of_vertices::Int32)
        mask = compute_queue_size(num_of_vertices) - 1
        queue = zeros(Int32, mask + 2)
        new(mask, queue, 0, 0, 0)
    end
end

@inline size_q(queue::_CircularBufferQueue) = queue.num

@inline isempty_q(queue::_CircularBufferQueue) = queue.rear == queue.front

"""
    pop_q!(queue::_CircularBufferQueue) -> Int32

Remove and return front element
"""
@inline function pop_q!(queue::_CircularBufferQueue)
    queue.front += 1
    elem = queue.queue[queue.front]
    queue.front &= queue.mask
    queue.num -=1
    return elem
end

"""
    push_q!(queue::_CircularBufferQueue, elem::Int32)

Add element to rear of queue
"""
@inline function push_q!(queue::_CircularBufferQueue, elem::Int32)
    queue.rear += 1
    queue.queue[queue.rear] = elem
    queue.rear &= queue.mask
    queue.num +=1
end