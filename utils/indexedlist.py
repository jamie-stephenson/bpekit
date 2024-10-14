class IndexedList:
    __slots__ = 'index', 'start'
    class Node:
        __slots__ = 'val', 'prev', 'next'
        def __init__(self, val, prev, next):
            self.val, self.prev, self.next = val, prev, next

        def delete(self):
            if self.prev is not None:
                self.prev.next = self.next
            if self.next is not None:
                self.next.prev = self.prev
            self.next = self.prev = None

    def __init__(self, l):
        self.index = {}
        l = iter(l)
        a = next(l)
        self.start = prev_node = IndexedList.Node(a, None, None)
        for b in l:
            prev_node.next = node = IndexedList.Node(b, prev_node, None)
            self.add_to_index((a, b), prev_node)
            a, prev_node = b, node

    def __iter__(self):
        node = self.start
        while node is not None:
            yield node
            node = node.next

    def update_index(self, node):  # Update index before/after node.
        if node.prev is not None:
            self.add_to_index((node.prev.val, node.val), node.prev)
        if node.next is not None:
            self.add_to_index((node.val, node.next.val), node)

    def add_to_index(self, pair, node):
        self.index.setdefault(pair, []).append(node)