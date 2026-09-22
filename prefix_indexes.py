"""Static exact-prefix indexes with a shared lexicographic top-k contract.

Keys are deduplicated Unicode strings. Rebuild to update a static index.
These indexes do not perform fuzzy or substring search.
"""
from array import array
from bisect import bisect_left
from collections import deque


class SortedPrefixIndex:
    def __init__(self, keys):
        self.keys = sorted(set(keys))

    def search(self, prefix, k=10):
        if k <= 0:
            return []
        start = bisect_left(self.keys, prefix)
        result = []
        for key in self.keys[start:start+k]:
            if not key.startswith(prefix):
                break
            result.append(key)
        return result


class RadixPrefixIndex:
    """Compressed edge-label trie. Each node stores a lexical key interval."""
    def __init__(self, keys):
        ordered = sorted(set(keys))
        self.keys = ordered
        # Temporary character trie, then iterative compression avoids recursion limits.
        root = {}
        for index, key in enumerate(ordered):
            node = root
            for char in key:
                node = node.setdefault(char, {})
            node[None] = index
        self.root = [[], 0, len(ordered)]
        stack = [(root, self.root)]
        while stack:
            source, target = stack.pop()
            for char in sorted(c for c in source if c is not None):
                edge, child = char, source[char]
                while None not in child and len(child) == 1:
                    char = next(iter(child)); edge += char; child = child[char]
                # Bounds are assigned in a later postorder pass.
                out = [[], child.get(None), None]
                target[0].append((edge, out)); stack.append((child, out))
        stack = [(self.root, False)]
        while stack:
            node, visited = stack.pop()
            if not visited:
                stack.append((node, True))
                stack.extend((child, False) for _, child in node[0])
            else:
                lows = [child[1] for _, child in node[0]]
                highs = [child[2] for _, child in node[0]]
                if node is self.root:
                    continue
                if node[1] is not None:
                    lows.append(node[1]); highs.append(node[1]+1)
                node[1], node[2] = min(lows), max(highs)

    def search(self, prefix, k=10):
        if k <= 0:
            return []
        node = self.root
        rest = prefix
        while rest:
            for edge, child in node[0]:
                if edge.startswith(rest):
                    return self.keys[child[1]:min(child[2], child[1]+k)]
                if rest.startswith(edge):
                    rest = rest[len(edge):]; node = child; break
            else:
                return []
        return self.keys[node[1]:min(node[2], node[1]+k)]


class DoubleArrayPrefixIndex:
    """Static base/check trie with typed arrays and lexical key intervals.

    Builder uses a temporary character trie; build peak is not resident index size.
    """
    def __init__(self, keys):
        self.keys = sorted(set(keys))
        self.codes = {c:i+1 for i,c in enumerate(sorted(set(''.join(self.keys))))}
        root = {}
        for index, key in enumerate(self.keys):
            node = root
            for char in key:
                node = node.setdefault(char, {})
            node[None] = index
        self.base = array('q', [0,0]); self.check = array('q', [-1,0])
        self.low = array('q', [0,0]); self.high = array('q', [0,len(self.keys)])
        queue = deque([(root,1)])
        parents = []
        next_free = 1
        while queue:
            source,state = queue.popleft()
            terminal = source.get(None)
            self.low[state] = terminal if terminal is not None else len(self.keys)
            self.high[state] = terminal+1 if terminal is not None else 0
            children = sorted(c for c in source if c is not None)
            if not children: continue
            codes = [self.codes[c] for c in children]
            while next_free < len(self.check) and self.check[next_free] != -1: next_free += 1
            offset = max(1,next_free-min(codes))
            while any(offset+c < len(self.check) and self.check[offset+c] != -1 for c in codes): offset += 1
            maximum = offset+max(codes)
            extend = maximum+1-len(self.check)
            if extend>0:
                self.base.extend([0]*extend);self.check.extend([-1]*extend)
                self.low.extend([0]*extend);self.high.extend([0]*extend)
            self.base[state]=offset
            for char,code in zip(children,codes):
                child = offset+code; self.check[child]=state
                parents.append((state,child));queue.append((source[char],child))
        for parent,child in reversed(parents):
            self.low[parent]=min(self.low[parent],self.low[child])
            self.high[parent]=max(self.high[parent],self.high[child])

    def search(self,prefix,k=10):
        if k<=0:return []
        state=1
        for char in prefix:
            code=self.codes.get(char)
            if code is None:return []
            next_state=self.base[state]+code
            if next_state>=len(self.check) or self.check[next_state]!=state:return []
            state=next_state
        return self.keys[self.low[state]:min(self.high[state],self.low[state]+k)]


class CharacterPrefixIndex:
    """Uncompressed character trie with the same static key-interval contract."""
    def __init__(self, keys):
        self.keys = sorted(set(keys))
        self.root = [{}, 0, len(self.keys)]
        for i, key in enumerate(self.keys):
            node = self.root
            for char in key:
                node = node[0].setdefault(char, [{}, i, i+1])
                node[2] = i+1

    def search(self, prefix, k=10):
        if k <= 0: return []
        node = self.root
        for char in prefix:
            node = node[0].get(char)
            if node is None: return []
        return self.keys[node[1]:min(node[2], node[1]+k)]


class TernaryPrefixIndex:
    """Iterative ternary search tree; each node has low/equal/high branches."""
    class Node:
        __slots__ = ('char','left','equal','right','low','high')
        def __init__(self, char, index):
            self.char=char;self.left=self.equal=self.right=None
            self.low=index;self.high=index+1

    def __init__(self, keys):
        self.keys=sorted(set(keys));self.root=None
        # Median-first insertion balances sibling comparisons for sorted input.
        work=[(0,len(self.keys))]
        while work:
            lo,hi=work.pop()
            if lo>=hi:continue
            i=(lo+hi)//2;key=self.keys[i]
            work.extend([(lo,i),(i+1,hi)])
            if not key:continue
            if self.root is None:self.root=self.Node(key[0],i)
            node=self.root;offset=0
            while True:
                char=key[offset]
                if char<node.char:
                    if node.left is None:node.left=self.Node(char,i)
                    node=node.left
                elif char>node.char:
                    if node.right is None:node.right=self.Node(char,i)
                    node=node.right
                else:
                    node.low=min(node.low,i);node.high=max(node.high,i+1)
                    offset+=1
                    if offset==len(key):break
                    if node.equal is None:node.equal=self.Node(key[offset],i)
                    node=node.equal

    def search(self,prefix,k=10):
        if k<=0:return []
        if not prefix:return self.keys[:k]
        node=self.root;offset=0
        while node:
            char=prefix[offset]
            if char<node.char:node=node.left
            elif char>node.char:node=node.right
            else:
                offset+=1
                if offset==len(prefix):return self.keys[node.low:min(node.high,node.low+k)]
                node=node.equal
        return []


class LinearPrefixIndex(SortedPrefixIndex):
    def search(self,prefix,k=10):
        if k<=0:return []
        result=[]
        for key in self.keys:
            if key.startswith(prefix):
                result.append(key)
                if len(result)==k:break
        return result
