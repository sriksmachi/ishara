
def generator(l, s):
    batches = len(l) // s
    for i in range(batches):
        yield (l[s*i:s+batches*i])
    r = len(l) % s
    yield r

g = generator([1,2,3,4,5], 2)
print(next(g))
print(next(g))
print(next(g))
